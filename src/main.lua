-------------------------------------------------------------------------------------
-- This is the main file to train the fusion model based on EMOTIC dataset
-------------------------------------------------------------------------------------

mini_dataset = false -- true/false useful for test-running the model to check if the variables have been set properly

require 'paths'
paths.dofile('vars.lua') -- Load the common variables' file 

-------------------------------------------
-- MODIFY PARAMETERS for different experiments 
-------------------------------------------
--IMGSizes.BODY_SIZE = 224 --default is 128 
--opt.nEpochs = 2         --uncomment to use nEpochs different than 21
--opt.nItersLR = 7        --uncomment to use nItersLR different than 7
--opt.criterion = 'cont'  --uncomment to use 'disc' or 'cont' else it will be 'joint'
--opt.LR = 0.001           --uncomment to use LR different than default 0.01 (fine-tuning)
--opt.reWeight = 1          --set to 1 for weighing at every batch, default is 0
--opt.optim = 'adam'       --uncomment to use adam, else will use 'sgd'
--BATCH_SIZE = 104          --uncomment to use size different than 52
--opt.body_last_layer = 'SAP'  --uncomment to use MP, default is SAP
--opt.image_last_layer = 'SAP' --uncomment to use MP, default is SAP
--opt.Wdisc = 1
--opt.Wcont = 1
--opt.model_type = 'B' --uncomment to use another type, default is 'BI'

saveTable(opt,paths.concat(opt.save,'options_commandline.txt'))

paths.dofile('train_test.lua')

-------------------------------------------------
-- Define the model type, intialize and test it 
------------------------------------------------
model={}
if opt.model_type == 'B' then 
    model = CreateEmotionModel_B(4,IMGSizes)    
    --check the model 
    assert(model:cuda():forward(torch.Tensor(BATCH_SIZE,3,IMGSizes.BODY_SIZE,IMGSizes.BODY_SIZE):cuda()))

elseif opt.model_type == 'I' then 
    model = CreateEmotionModel_I(4,IMGSizes)
    --check the model
    assert(model:cuda():forward(torch.Tensor(BATCH_SIZE,3,IMGSizes.INPUT_LARGEST_SIZE,IMGSizes.INPUT_LARGEST_SIZE):cuda()))
    
elseif opt.model_type == 'BI' then 
    model = CreateEmotionModel_BI(4,IMGSizes)
    --check the model
assert(model:cuda():forward({torch.Tensor(BATCH_SIZE,3,IMGSizes.INPUT_LARGEST_SIZE,IMGSizes.INPUT_LARGEST_SIZE):cuda(),torch.Tensor(BATCH_SIZE,3,IMGSizes.BODY_SIZE,IMGSizes.BODY_SIZE):cuda()}))

end
model:cuda()

--------------------
-- Define Criterion
--------------------
criterion = DefineCriterion()
criterion:cuda()

---------------
-- Timer Setup
---------------
timer = torch.Timer()
dataTimer = torch.Timer()

-----------------------
-- Log File Definition
-----------------------
print("========= Creating Log Files =======")
logtrainname = 'train_' .. opt.append .. '.log'
trainLogger = optim.Logger(paths.concat(opt.save, logtrainname))
trainLogger:setNames{'Epoch','train_loss/batch','avgauc','avg-AP'}
logtestname = 'test_' .. opt.append .. '.log'
testLogger = optim.Logger(paths.concat(opt.save, logtestname))
testLogger:setNames{'Epoch','test_loss/batch','test_avgauc','test_globauc','avg-AP',Nconcepts[1],Nconcepts[2],Nconcepts[3]}
logvalname = 'val_' .. opt.append .. '.log'
valLogger = optim.Logger(paths.concat(opt.save, logvalname))
valLogger:setNames{'Epoch','val_loss/batch','val_avgauc','val_globauc','avg-AP',Nconcepts[1],Nconcepts[2],Nconcepts[3]}
print("Saving logs to: ");
print(logtrainname);
print(logtestname);
print(logvalname);
print("==================================")

--get the parameters
if opt.freezeWeights == 1 then 
    parameters, gradParameters = model:parameters() --comment only when using the nn.Optim
else 
    parameters, gradParameters = model:getParameters() --comment only when using the nn.Optim
end

------------------------------
-- Setup optimization State
------------------------------
if opt.optim == 'sgd' then 
    if opt.freezeWeights == 1 then --if the weights of lower layers are freezed, 
                                   --while using optim.sgd, then weight decay has to be set to 0
        opt.weightDecay = 0 
        opt.momentum = 0
    end
    
    optimState = {
        learningRate = opt.LR,
        learningRateDecay = 0.0,
        momentum = opt.momentum,
        dampening = 0.0,
        weightDecay = opt.weightDecay
    }
elseif opt.optim == 'adam' then 
    optimState = {
        learningRate = opt.LR,
        learningRateDecay = 0.0,
        weightDecay = opt.weightDecay,
        momentum = 0,
        beta1 = 0.9,
        beta2 = 0.999,
        epsilon = 1e-8
    }
end

--optimator = nn.Optim(model, optimState) --uncomment only when using the nn.Optim

-- Main loop
epoch = opt.epochNumber
for main_epoch = opt.epochNumber,opt.nEpochs do
	epoch = main_epoch
    if main_epoch%opt.nItersLR == 0 then
        optimState = {
            learningRate = math.max(0.0000001,optimState.learningRate/10),
            learningRateDecay = 0.0,
            momentum = opt.momentum,
            dampening = 0.0,
            weightDecay = opt.weightDecay
        }
    end
    --[[ train ]]
    train_both(epoch)

    --[[ val ]]
--    local test_flag = test_both(1, epoch, best_epoch) -- 1 for validation
    test_both(1, epoch, best_epoch) -- 1 for validation
    --[[ test ]]
    --if test_flag == 1 then  -- if the validation error is better, then test
    test_both(0, epoch, best_epoch) 
	--end
end

--find the best epoch and save a text file (0_best_epoch.log) with the description in the same folder
findBestEpoch_joint(logvalname, opt.save, opt.nEpochs)
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
