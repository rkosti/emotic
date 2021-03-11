-- Copyright 2004-present Facebook. All Rights Reserved.

local pl = require('pl.import_into')()

-- from fblualib/fb/util/data.lua , copied here because fblualib is not rockspec ready yet.
-- deepcopy routine that assumes the presence of a 'clone' method in user
-- data should be used to deeply copy. This matches the behavior of Torch
-- tensors.
local function deepcopy(x)
    local typename = type(x)
    if typename == "userdata" then
        return x:clone()
    end
    if typename == "table" then
        local retval = { }
        for k,v in pairs(x) do
            retval[deepcopy(k)] = deepcopy(v)
        end
        return retval
    end
    return x
end

local Optim, parent = torch.class('nn.Optim')


-- Returns weight parameters and bias parameters and associated grad parameters
-- for this module. Annotates the return values with flag marking parameter set
-- as bias parameters set
function Optim.weight_bias_parameters(module)
    local weight_params, bias_params
    if module.weight then
        weight_params = {module.weight, module.gradWeight}
        weight_params.is_bias = false
    end
    if module.bias then
        bias_params = {module.bias, module.gradBias}
        bias_params.is_bias = true
    end
    return {weight_params, bias_params}
end

-- The regular `optim` package relies on `getParameters`, which is a
-- beastly abomination before all. This `optim` package uses separate
-- optim state for each submodule of a `nn.Module`.
function Optim:__init(model, optState, checkpoint_data)
    assert(model)
    assert(checkpoint_data or optState)
    assert(not (checkpoint_data and optState))

    self.model = model
    self.modulesToOptState = {}
    -- Keep this around so we update it in setParameters
    self.originalOptState = optState

    -- Each module has some set of parameters and grad parameters. Since
    -- they may be allocated discontinuously, we need separate optState for
    -- each parameter tensor. self.modulesToOptState maps each module to
    -- a lua table of optState clones.
    if not checkpoint_data then
       -- self.model:for_each(function(module)
        self.model:apply(function(module)
            self.modulesToOptState[module] = { }
            local params = self.weight_bias_parameters(module)
            -- expects either an empty table or 2 element table, one for weights
            -- and one for biases
            assert(pl.tablex.size(params) == 0 or pl.tablex.size(params) == 2)
            for i, _ in ipairs(params) do
                self.modulesToOptState[module][i] = deepcopy(optState)
                if params[i] and params[i].is_bias then
                    -- never regularize biases
                    self.modulesToOptState[module][i].weightDecay = 0.0
                end
                self.modulesToOptState[module][i].name = module.name;
            end
            assert(module)
            assert(self.modulesToOptState[module])
        end)
    else
        local state = checkpoint_data.optim_state
        local modules = {}
        self.model:for_each(function(m) table.insert(modules, m) end)
        assert(pl.tablex.compare_no_order(modules, pl.tablex.keys(state)))
        self.modulesToOptState = state
    end
end

function Optim:save()
    return {
        optim_state = self.modulesToOptState
    }
end

local function _type_all(obj, t)
    for k, v in pairs(obj) do
        if type(v) == 'table' then
            _type_all(v, t)
        else
            local tn = torch.typename(v)
            if tn and tn:find('torch%..+Tensor') then
                obj[k] = v:type(t)
            end
        end
    end
end

function Optim:type(t)
    self.model:for_each(function(module)
        local state= self.modulesToOptState[module]
        assert(state)
        _type_all(state, t)
    end)
end

local function get_device_for_module(mod)
   local dev_id = nil
   for name, val in pairs(mod) do
       if torch.typename(val) == 'torch.CudaTensor' then
           local this_dev = val:getDevice()
           if this_dev ~= 0 then
               -- _make sure the tensors are allocated consistently
               assert(dev_id == nil or dev_id == this_dev)
               dev_id = this_dev
           end
       end
   end
   return dev_id -- _may still be zero if none are allocated.
end

local function on_device_for_module(mod, f)
    local this_dev = get_device_for_module(mod)
    if this_dev ~= nil then
        return cutorch.withDevice(this_dev, f)
    end
    return f()
end

function Optim:On_device_for_module(mod, f)
    local this_dev = get_device_for_module(mod)
    if this_dev ~= nil then
        return cutorch.withDevice(this_dev, f)
    end
    return f()
end




function Optim:optimize(optimMethod, inputs, targets, criterion)
    assert(optimMethod)
    assert(inputs)
    assert(targets)
    assert(criterion)
    assert(self.modulesToOptState)

    self.model:zeroGradParameters()
    local output = self.model:forward(inputs)


   
    local err = criterion:forward(output, targets)
    local df_do = criterion:backward(output, targets)    
    self.model:backward(inputs, df_do)
   --[[
     local err = criterion:forward(output:float(), targets:float())
    local df_do = criterion:backward(output:float(), targets:float())    
    self.model:backward(inputs, df_do:cuda())
 ]]--
    -- We'll set these in the loop that iterates over each module. Get them
    -- out here to be captured.
    local curGrad
    local curParam
    local function fEvalMod(x)
        return err, curGrad
    end

    for curMod, opt in pairs(self.modulesToOptState) do
        on_device_for_module(curMod, function()
            local curModParams = self.weight_bias_parameters(curMod)
            -- expects either an empty table or 2 element table, one for weights
            -- and one for biases
            assert(pl.tablex.size(curModParams) == 0 or
                   pl.tablex.size(curModParams) == 2)
            if curModParams then
                for i, tensor in ipairs(curModParams) do
                    if curModParams[i] then
                        -- expect param, gradParam pair
                        curParam, curGrad = table.unpack(curModParams[i])
                        assert(curParam and curGrad)
--print(epoch)
                        optimMethod(fEvalMod, curParam, opt[i])
                    end
                end
            end
        end)
    end

    return err, output
end


function Optim:optimize_wEuc(optimMethod, inputs, targets, criterion)
    assert(optimMethod)
    assert(inputs)
    assert(targets)
    assert(criterion)
    assert(self.modulesToOptState)

    self.model:zeroGradParameters()
    local output = self.model:forward(inputs)


   --print(torch.type(output))
    local err = criterion:forward(output:float(), targets:float())
    local df_do = criterion:backward(output:float(), targets:float())    
   
   --print(torch.type(output))
   --print(torch.type(df_do))
   self.model:backward(inputs, df_do:cuda())
   --[[
     local err = criterion:forward(output:float(), targets:float())
    local df_do = criterion:backward(output:float(), targets:float())    
    self.model:backward(inputs, df_do:cuda())
 ]]--
    -- We'll set these in the loop that iterates over each module. Get them
    -- out here to be captured.
    local curGrad
    local curParam
    local function fEvalMod(x)
        return err, curGrad
    end

    for curMod, opt in pairs(self.modulesToOptState) do
        on_device_for_module(curMod, function()
            local curModParams = self.weight_bias_parameters(curMod)
            -- expects either an empty table or 2 element table, one for weights
            -- and one for biases
            assert(pl.tablex.size(curModParams) == 0 or
                   pl.tablex.size(curModParams) == 2)
            if curModParams then
                for i, tensor in ipairs(curModParams) do
                    if curModParams[i] then
                        -- expect param, gradParam pair
                        curParam, curGrad = table.unpack(curModParams[i])
                        assert(curParam and curGrad)
--print(epoch)
                        optimMethod(fEvalMod, curParam, opt[i])
                    end
                end
            end
        end)
    end

    return err, output
end

function Optim:optimize_wmEuc(optimMethod, inputs, targets, criterion)
    assert(optimMethod)
    assert(inputs)
    --assert(targets[1])
    --assert(targets[2])
    assert(criterion)
    assert(self.modulesToOptState)

    self.model:zeroGradParameters()
    local output = self.model:forward(inputs)
    
    targets[1]:float();
    targets[2]:float();
    
   --
    local err = criterion:forward(output, targets)    
    
    
    
    local df_do = criterion:backward({output[1]:float(),output[2]:float()}, targets)
    --local df_do = criterion:backward(output, targets)
      
   self.model:backward(inputs, {df_do[1]:cuda(),df_do[2]:cuda()})
   --[[
     local err = criterion:forward(output:float(), targets:float())
    local df_do = criterion:backward(output:float(), targets:float())    
    self.model:backward(inputs, df_do:cuda())
 ]]--
    -- We'll set these in the loop that iterates over each module. Get them
    -- out here to be captured.
    local curGrad
    local curParam
    local function fEvalMod(x)
        return err, curGrad
    end

    for curMod, opt in pairs(self.modulesToOptState) do
        on_device_for_module(curMod, function()
            local curModParams = self.weight_bias_parameters(curMod)
            -- expects either an empty table or 2 element table, one for weights
            -- and one for biases
            assert(pl.tablex.size(curModParams) == 0 or
                   pl.tablex.size(curModParams) == 2)
            if curModParams then
                for i, tensor in ipairs(curModParams) do
                    if curModParams[i] then
                        -- expect param, gradParam pair
                        curParam, curGrad = table.unpack(curModParams[i])
                        assert(curParam and curGrad)
--print(epoch)
                        optimMethod(fEvalMod, curParam, opt[i])
                    end
                end
            end
        end)
    end

    return err, output
end


function Optim:optimizeKL(optimMethod, inputs, targets, criterion)
    assert(optimMethod)
    assert(inputs)
    assert(targets)
    assert(criterion)
    assert(self.modulesToOptState)

    self.model:zeroGradParameters()
    local output = self.model:forward(inputs)


    criterion:cuda();
    --print('Targets ' .. torch.type(targets))
    --print('Input ' .. torch.type(output))
    
    local err = criterion:forward(output, targets)
    local df_do = criterion:backward(output, targets)    
    self.model:backward(inputs, df_do)
   --[[
     local err = criterion:forward(output:float(), targets:float())
    local df_do = criterion:backward(output:float(), targets:float())    
    self.model:backward(inputs, df_do:cuda())
 ]]--
    -- We'll set these in the loop that iterates over each module. Get them
    -- out here to be captured.
    local curGrad
    local curParam
    local function fEvalMod(x)
        return err, curGrad
    end

    for curMod, opt in pairs(self.modulesToOptState) do
        on_device_for_module(curMod, function()
            local curModParams = self.weight_bias_parameters(curMod)
            -- expects either an empty table or 2 element table, one for weights
            -- and one for biases
            assert(pl.tablex.size(curModParams) == 0 or
                   pl.tablex.size(curModParams) == 2)
            if curModParams then
                for i, tensor in ipairs(curModParams) do
                    if curModParams[i] then
                        -- expect param, gradParam pair
                        curParam, curGrad = table.unpack(curModParams[i])
                        assert(curParam and curGrad)
--print(epoch)
                        optimMethod(fEvalMod, curParam, opt[i])
                    end
                end
            end
        end)
    end

    return err, output
end



function Optim:setParameters(newParams)
    assert(newParams)
    assert(type(newParams) == 'table')
    local function splice(dest, src)
        for k,v in pairs(src) do
            dest[k] = v
        end
    end

    splice(self.originalOptState, newParams)
    for _,optStates in pairs(self.modulesToOptState) do
        for i,optState in pairs(optStates) do
            assert(type(optState) == 'table')
            splice(optState, newParams)
        end
    end
end


local function isField(t,s)
  if t == nil then return false end
  local t = t
  for key in s:gmatch('[^.]+') do
    if t[ key ] == nil then return false end
    t = t[ key ]
  end
  return true
end


function Optim:CleanModule(layerName,zeroes)
  local bLayerFound=false;
 for curMod, opt in pairs(self.modulesToOptState) do
 on_device_for_module(curMod, function()
    local params = self.weight_bias_parameters(curMod)
    for i, _ in ipairs(params) do  
        if isField(self.modulesToOptState[curMod][i],'name') then          
           local b = string.match(self.modulesToOptState[curMod][i].name,layerName)	
           if b~= nil then                          
            -- print(layerName);
             bLayerFound=true;
            if i == 1 then          
               local ii=1;
               --print(#zeroes);
                for ind, ii in ipairs(zeroes) do
                --print(ii)--this will return the filters to be zeroed		
                self.modulesToOptState[curMod][i].dfdx:select(1,ii):zero();                
               end
            elseif i ==2 then
              local ii=1;
              for ind, ii in ipairs(zeroes) do
                              self.modulesToOptState[curMod][i].dfdx[ii]=0;
              end
            end
          end
        end
    end
  end)
 end
 return bLayerFound;
end




















