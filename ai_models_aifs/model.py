# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True' 
os.environ['ANEMOI_INFERENCE_NUM_CHUNKS']='16'
import torch

import logging
from collections import defaultdict
import earthkit.data as ekd
import earthkit.regrid as ekr
from earthkit.data.indexing.fieldlist import FieldArray
from anemoi.transform.grids.unstructured import UnstructuredGridFieldList
from anemoi.inference.checkpoint import Checkpoint

import numpy as np
from ai_models.model import Model

LOG = logging.getLogger(__name__)
def rename(f, _, metadata):
    return metadata["param"]

class AIFS(Model):

    download_url = "https://huggingface.co/ecmwf/aifs-single-1.1/resolve/main/{file}"
    download_files = ["aifs-single-mse-1.1.ckpt"]

    # Input
    area = [90, 0, -90, 360 - 0.25]
    grid = [0.25, 0.25]

    param_sfc = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor"]
    param_level_pl = (
        ["z", "t", "u", "v", "w", "q"],
        [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
    )
    param_level_sol = (
        ["vsw", "sot"],
        [1, 2],
    )

    mapping = {'sot_1': 'stl1', 'sot_2': 'stl2', 'vsw_1': 'swvl1','vsw_2': 'swvl2'}
    lagged = (-6, 0)

    # Output

    expver = "aifs"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hour_steps = 6
        self.checkpoint_path = os.path.join(self.assets, self.download_files[0])
        self.ckpt = Checkpoint(self.checkpoint_path)
        self.metadata = self.ckpt._metadata
        self.template_field = {} # for writing output

    def get_init_time(self):
        init_time = self.all_fields.order_by(valid_datetime="descending")[0].datetime()["base_time"]
        return init_time

    def get_source(self):
        source = UnstructuredGridFieldList.from_values(
            latitudes=self.metadata.latitudes,
            longitudes=self.metadata.longitudes,
        )
        return source

    def get_forcing(self, variables, dates):
        source = self.get_source()
        ds = ekd.from_source("forcings", source, date=dates, param=variables)
        ds = FieldArray([f.clone(name=rename) for f in ds])
        forcing = ds.order_by(name=variables, valid_datetime="ascending")
        forcings = forcing.to_numpy(dtype=np.float32, flatten=True).reshape(len(variables), len(dates), -1)
        return forcings

    def get_input(self, init_time):
        init_time_str = init_time.strftime('%Y-%m-%dT%H:%M:%S')
        fields_pl = self.fields_pl
        param_pl, level_pl = self.param_level_pl
        fields_pl = fields_pl.sel(param=param_pl, level=level_pl)
        fields_pl = fields_pl.order_by('valid_datetime', param=param_pl, level=level_pl)

        fields_sfc = self.fields_sfc
        fields_sfc = fields_sfc.sel(param=self.param_sfc)
        fields_sfc = fields_sfc.order_by('valid_datetime', param=self.param_sfc)

        fields_sol = self.fields_sol
        param_sol, level_sol = self.param_level_sol
        fields_sol = fields_sol.sel(param=param_sol, level=level_sol)
        fields_sol = fields_sol.order_by('valid_datetime', param=param_sol, level=level_sol)

        output_tensor_index_to_variable = self.ckpt.output_tensor_index_to_variable
        fields = defaultdict(list)
        for f in fields_sfc + fields_sol + fields_pl:
            values = ekr.interpolate(f.to_numpy(), {"grid": (0.25, 0.25)}, {"grid": "N320"})
            param = f.metadata("param")
            levtype = f.metadata("levtype")
            name = f"{param}_{f.metadata('level')}" if levtype != 'sfc' else param
            param = self.mapping.get(name, name)
            if param in output_tensor_index_to_variable.values() and f.metadata("valid_datetime") == init_time_str:
                self.template_field[param] = f
            fields[param].append(values)
        for param, values in fields.items():
            fields[param] = np.stack(values)

        input_fields = {}
        input_fields.update(fields)

        timestep = self.metadata.timestep
        hist_time = init_time - timestep
        dates = [hist_time, init_time]

        constant_forcing_variables = self.metadata.computed_constant_forcings[1]
        forcings = self.get_forcing(constant_forcing_variables, dates)
        for name, forcing in zip(constant_forcing_variables, forcings):
            input_fields[name] = forcing

        time_dependent_forcing_variables = self.metadata.computed_time_dependent_forcings[1]
        forcings = self.get_forcing(time_dependent_forcing_variables, dates)
        for name, forcing in zip(time_dependent_forcing_variables, forcings):
            input_fields[name] = forcing

        input_tensor_numpy = np.full(
            shape=(
                self.ckpt.multi_step_input,
                self.ckpt.number_of_input_features,
                self.metadata.latitudes.size,
            ),
            fill_value=np.nan,
            dtype=np.float32,
        )

        variable_to_input_tensor_index = self.ckpt.variable_to_input_tensor_index
        for var, field in input_fields.items():
            i = variable_to_input_tensor_index[var]
            input_tensor_numpy[:, i] = field

        return input_tensor_numpy

    def load_model(self, checkpoint_file):
        model = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
        model.to(self.device)
        model.eval()

        return model

    def run(self):

        init_time = self.get_init_time()
        input_tensor_numpy = self.get_input(init_time)

        torch.set_grad_enabled(False)
        model = self.load_model(self.checkpoint_path)

        # Create pytorch input tensor
        input_tensor_torch = torch.from_numpy(np.swapaxes(input_tensor_numpy, -2, -1)[np.newaxis, ...]).to(self.device)
        total_step = self.lead_time // self.hour_steps

        time_dependent_forcing_mask, time_dependent_forcing_variables = self.metadata.computed_time_dependent_forcings
        output_tensor_index_to_variable = self.ckpt.output_tensor_index_to_variable

        with self.stepper(6) as stepper:
            for i in range(total_step):
                step = (i + 1) * self.hour_steps
                date = init_time + self.metadata.timestep * (i + 1)
                # Predict next state of atmosphere
                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    y_pred = model.predict_step(input_tensor_torch, fcstep=i, step=step, date=date)

                # Detach tensor and squeeze (should we detach here?)
                output = np.squeeze(y_pred.cpu().numpy())
                # Update state
                for j in range(output.shape[1]):
                    param = output_tensor_index_to_variable[j]
                    values = ekr.interpolate(output[:, j], {"grid": "N320"}, {"grid": (0.25, 0.25)})
                    if param in self.template_field:
                        self.write(values, template=self.template_field[param], step=step)
                    else:
                        template = self.template_field['sp']
                        new_template = template.clone(metadata=template.metadata().override(shortName=param))
                        self.write(values, template=new_template, step=step)

                stepper(i, step)

                if i < total_step - 1:
                    prognostic_output_mask = self.ckpt.prognostic_output_mask
                    prognostic_input_mask = self.ckpt.prognostic_input_mask

                    prognostic_fields = y_pred[..., prognostic_output_mask]  # Get new predicted values
                    input_tensor_torch = input_tensor_torch.roll(-1, dims=1)  # Roll the tensor in the multi_step_input dimension
                    input_tensor_torch[:, -1, :, prognostic_input_mask] = (
                        prognostic_fields  # Add new values to last 'multi_step_input' row
                    )

                    forcings = self.get_forcing(time_dependent_forcing_variables, [date])
                    forcings = np.squeeze(forcings, axis=1)  # Drop the dates dimension
                    forcings = np.swapaxes(forcings[np.newaxis, np.newaxis, ...], -2, -1)  # shape: (1, 1, values, variables)
                    forcings = torch.from_numpy(forcings).to(self.device)  # Copy to device
                    input_tensor_torch[:, -1, :, time_dependent_forcing_mask] = forcings  # Copy forcings to last 'multi_step_input' row
