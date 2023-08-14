# GRU_regression_gas-sensor
train process  in baseline.py

model test in eval folder 'test.py'\
change model floder  "GRU + beta"  file
- add "learnable_variable = nn.Parameter(torch.Tensor(1))"
- add "self.bweight1 = learnable_variable"
  
- add self.bweight1.data = torch.clamp(self.bweight1.data, 1, 10)
- change "out = out - (5 * x_r)"  to "out = out - (self.bweight1 * x_r)"

File names with "var_par" in them are the results of training using learnable variables.