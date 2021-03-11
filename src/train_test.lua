--********************************************************************
-- TRAINING 
--********************************************************************

function train_both() 
    if opt.criterion == 'joint' then
        print("==> Reseting Euclidean Weights for Loss function")
        criterion.criterions[2].weights:copy(torch.Tensor(gClass_weights))
    elseif  opt.criterion == 'disc' then
        print("==> Reseting Euclidean Weights for Loss function")
        criterion.weights:copy(torch.Tensor(gClass_weights))
    end
    
    loss_epoch = 0
    batchNumber = 0

    ---------------------
    -- Memory allocation
    ---------------------
    print('==> allocating minibatch memory')
    if opt.model_type == 'B' then 
        x_body = torch.Tensor(BATCH_SIZE, 3, IMGSizes.BODY_SIZE, IMGSizes.BODY_SIZE)       
    elseif opt.model_type == 'I' then
        x_img = torch.Tensor(BATCH_SIZE, 3, IMGSizes.INPUT_LARGEST_SIZE, IMGSizes.INPUT_LARGEST_SIZE)
    elseif opt.model_type == 'BI' then 
        x_img = torch.Tensor(BATCH_SIZE, 3, IMGSizes.INPUT_LARGEST_SIZE, IMGSizes.INPUT_LARGEST_SIZE)
        x_body = torch.Tensor(BATCH_SIZE, 3, IMGSizes.BODY_SIZE, IMGSizes.BODY_SIZE)       
    end
    
    print('==> allocating global-test memory')
    if opt.criterion == 'joint' then 
        yt_disc         = torch.Tensor(BATCH_SIZE, IMGSizes.NUM_CLASSES):fill(0)
        yt_cont         = torch.Tensor(BATCH_SIZE, IMGSizes.NUM_CLASSES_CONTINUOUS):fill(0)
        ygt_train_cont  = torch.Tensor(#TrainData, IMGSizes.NUM_CLASSES_CONTINUOUS):fill(0)
        yest_train_cont = torch.Tensor(#TrainData, IMGSizes.NUM_CLASSES_CONTINUOUS):fill(0)
        ygt_train_disc  = torch.Tensor(#TrainData, IMGSizes.NUM_CLASSES):fill(0)
        yest_train_disc = torch.Tensor(#TrainData, IMGSizes.NUM_CLASSES):fill(0)
        
    elseif opt.criterion == 'cont' then
        yt_cont         = torch.Tensor(BATCH_SIZE, IMGSizes.NUM_CLASSES_CONTINUOUS):fill(0)
        ygt_train_cont  = torch.Tensor(#TrainData, IMGSizes.NUM_CLASSES_CONTINUOUS):fill(0)
        yest_train_cont = torch.Tensor(#TrainData, IMGSizes.NUM_CLASSES_CONTINUOUS):fill(0)
        
    elseif opt.criterion == 'disc' then
        yt_disc         = torch.Tensor(BATCH_SIZE, IMGSizes.NUM_CLASSES):fill(0)
        ygt_train_disc  = torch.Tensor(#TrainData, IMGSizes.NUM_CLASSES):fill(0)
        yest_train_disc = torch.Tensor(#TrainData, IMGSizes.NUM_CLASSES):fill(0)        
    end
    print '================================='   
  
    -------------------------
    -- Training Variables
    -------------------------
    print("==> Training (Uniform sampling): epoch # " .. epoch .. ' [batchSize = ' .. BATCH_SIZE .. ']')
    local tm = torch.Timer() 
    local iDataSize = #TrainData
    local shuffle  = torch.randperm(#TrainData) --we will always get the same randomization    
    epochSize = torch.round(iDataSize/BATCH_SIZE +0.5)
    batchNumber = 0
    
    ------------------
    -- Training Mode
    ------------------
    model:training() 
    local iTrainLimit = epochSize;--Set to iDataSize for pure epoch concept
    for t = 1, iDataSize, BATCH_SIZE do  --for pure epoch
        if (t + BATCH_SIZE - 1) > iDataSize then
            break
        end
        local idx = 1
        for i = t,t+BATCH_SIZE-1 do                    
            indAnn = shuffle[i]
            filename = BASE_IMAGE_FOLDER .. '/' .. TrainData[indAnn].folder .. '/' ..  TrainData[indAnn].filename
            input = image.load(filename)
            local ImgPatches = GetImagePatches(filename,TrainData[indAnn],IMGSizes)
            
            ----------
            -- Inputs
            ----------
            if opt.model_type == 'B' then 
                x_body[idx] = ImgPatches[2]
            elseif opt.model_type == 'I' then
                x_img[idx] = ImgPatches[1] 
            elseif opt.model_type == 'BI' then 
                x_img[idx] = ImgPatches[1] 
                x_body[idx] = ImgPatches[2]
            end
            
            ----------
            --Labels
            ----------
             if opt.criterion == 'joint' then 
                yt_disc[idx]:copy(TrainLabels[indAnn])
                yt_cont[idx]:copy(TrainContinuous[indAnn])
            elseif opt.criterion == 'cont' then
                yt_cont[idx]:copy(TrainContinuous[indAnn])
            elseif opt.criterion == 'disc' then
                yt_disc[idx]:copy(TrainLabels[indAnn])        
            end
            idx = idx + 1 
        end

        -------------------
        --DATA AUGMENTATION
        -------------------
        if opt.dataAugment == 1 and epoch%2 ~= 0 then
        -- keep inputs in float to avoid preprocess 'accidents'
            x_img = x_img:float()
            x_body = x_body:float()

            for ind_imgs=1,x_img:size(1) do 
                x_img[ind_imgs] = preprocess(x_img[ind_imgs])
                x_body[ind_imgs] = preprocess(x_body[ind_imgs])
            end

            -- convert inputs back to CUDA
            x_img = x_img:cuda()
            x_body = x_body:cuda()
        end

        
        -------------------
        -- Predictions
        -------------------
        if opt.criterion == 'joint' then 
            if opt.model_type == 'BI' then 
                preds = trainBatch({x_img:cuda(),x_body:cuda()}, {yt_cont:cuda(),yt_disc:cuda()})
            elseif opt.model_type == 'B' then 
                preds = trainBatch(x_body:cuda(), {yt_cont:cuda(),yt_disc:cuda()})
            elseif opt.model_type == 'I' then 
                preds = trainBatch(x_img:cuda(), {yt_cont:cuda(),yt_disc:cuda()}) 
            end

        elseif opt.criterion == 'cont' then
            if opt.model_type == 'BI' then 
                preds = trainBatch({x_img:cuda(),x_body:cuda()}, yt_cont:cuda())
            elseif opt.model_type == 'B' then 
                preds = trainBatch(x_body:cuda(), yt_cont:cuda())
            elseif opt.model_type == 'I' then 
                preds = trainBatch(x_img:cuda(), yt_cont:cuda()) 
            end
        elseif opt.criterion == 'disc' then
            if opt.model_type == 'BI' then 
                preds = trainBatch({x_img:cuda(),x_body:cuda()}, yt_disc:cuda())
            elseif opt.model_type == 'B' then 
                preds = trainBatch(x_body:cuda(), yt_disc:cuda())
            elseif opt.model_type == 'I' then 
                preds = trainBatch(x_img:cuda(), yt_disc:cuda()) 
            end
        end

        --store outcomes globally
        local idx2 = 1
        for indAnn = t,t+BATCH_SIZE-1 do          
             if opt.criterion == 'joint' then 
                
                --Predictions
                yest_train_cont[t+idx2-1]:copy(preds[1][idx2])
                yest_train_disc[t+idx2-1]:copy(preds[2][idx2])                
                --Labels
                ygt_train_cont[t+idx2-1]:copy(yt_cont[idx2])
                ygt_train_disc[t+idx2-1]:copy(yt_disc[idx2])
                
            elseif opt.criterion == 'cont' then
                
                --Predictions
                yest_train_cont[t+idx2-1]:copy(preds[idx2])                
                --Labels
                ygt_train_cont[t+idx2-1]:copy(yt_cont[idx2])
                
            elseif opt.criterion == 'disc' then
                
                --Predictions
                yest_train_disc[t+idx2-1]:copy(preds[idx2])                
                --Labels
                ygt_train_disc[t+idx2-1]:copy(yt_disc[idx2])

            end  
            
            idx2= idx2 + 1
        end       
    end    
    
    local loss_batch = loss_epoch / epochSize 
    model:clearState()
    torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model)
    torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
    
    if opt.criterion == 'joint' then 
       
        -- Curve Evaluation
        print("==> TRAIN-CURVE EVALUATION ")
        auc_avg,gauc,ap_avg,gftap = CurveEvaluation(yest_train_disc, ygt_train_disc,gClass_naming)
        print("===========================")

        print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'                          
                              .. 'average loss (per batch): %.2f \n avg AUC: %.2f \t avg-PR: %.2f', epoch, tm:time().real, loss_batch,auc_avg,ap_avg))
        print('\n')

        ---------------------------------
        -- Save Models, Logs, predictions
        ---------------------------------
        trainLogger:add{epoch,loss_batch,auc_avg,ap_avg}    

        matfilename = paths.concat(opt.save, 'train_predictions_disc_' .. epoch .. '.mat')
        mat.save(matfilename , yest_train_disc:double())
        matfilename = paths.concat(opt.save, 'train_gt_' .. epoch .. '.mat')
        mat.save(matfilename , ygt_train_disc:double())

        matfilename = paths.concat(opt.save, 'train_predictions_cont_' .. epoch .. '.mat')
        mat.save(matfilename , yest_train_cont:double())
        matfilename = paths.concat(opt.save, 'train_gt_cont_' .. epoch .. '.mat')
        mat.save(matfilename , ygt_train_cont:double())
        
    elseif opt.criterion == 'cont' then 
        print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'    
                              .. 'average loss (per batch): %.2f', epoch, tm:time().real, loss_batch))
        print('\n')
        ---------------------------------
        -- Save Models, Logs, predictions
        ---------------------------------
        trainLogger:add{epoch,loss_batch,'NA','NA'}  
        matfilename = paths.concat(opt.save, 'train_predictions_cont_' .. epoch .. '.mat')
        mat.save(matfilename , yest_train_cont:double())
        matfilename = paths.concat(opt.save, 'train_gt_cont_' .. epoch .. '.mat')
        mat.save(matfilename , ygt_train_cont:double())
    elseif opt.criterion == 'disc' then 
        
        -- Curve Evaluation
        print("==> TRAIN-CURVE EVALUATION ")
        auc_avg,gauc,ap_avg,gftap = CurveEvaluation(yest_train_disc, ygt_train_disc,gClass_naming)
        print("===========================")

        print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'                          
                              .. 'average loss (per batch): %.2f \n avg AUC: %.2f \t avg-PR: %.2f', epoch, tm:time().real, loss_batch,auc_avg,ap_avg))
        print('\n')

        ---------------------------------
        -- Save Models, Logs, predictions
        ---------------------------------
        trainLogger:add{epoch,loss_batch,auc_avg,ap_avg}    

        matfilename = paths.concat(opt.save, 'train_predictions_disc_' .. epoch .. '.mat')
        mat.save(matfilename , yest_train_disc:double())
        matfilename = paths.concat(opt.save, 'train_gt_' .. epoch .. '.mat')
        mat.save(matfilename , ygt_train_disc:double())

    end
    
    collectgarbage()
end 


function trainBatch(inputs, labels)
   --cutorch.synchronize()
   local dataLoadingTime = dataTimer:time().real
   timer:reset()

   -- transfer over to GPU
--   inputs:resize(inputsCPU:size()):copy(inputsCPU)
--   labels:resize(labelsCPU:size()):copy(labelsCPU)

    --------------------------------------
    -- Reweight the labels as per the batch.
    if opt.criterion == 'joint' then
        if opt.reWeight > 0 then 
            local class_weights = getWeightNormalization(labels[2],IMGSizes.NUM_CLASSES,opt.normWeights,opt.normFactor)      
            criterion.criterions[2].weights:copy(class_weights)                 
        end 
    elseif opt.criterion == 'disc' then
        if opt.reWeight > 0 then 
            local class_weights = getWeightNormalization(labels,IMGSizes.NUM_CLASSES,opt.normWeights,opt.normFactor)      
            criterion.weights:copy(class_weights)                 
        end 
    end

   local err = 0 
   local outputs = {}
--[[
    print('Inputdata size')
    print(#inputs)
    print('LabelsData size')
    print(#labels)
]]--
    
    if opt.optim == 'sgd' then
        if opt.freezeWeights == 1 then 
            for ind=1,#parameters do 
                feval = function(x)
                    model:zeroGradParameters()
                    outputs = model:forward(inputs)
                    err = criterion:forward(outputs, labels)
                    local gradOutputs = criterion:backward(outputs, labels)
                    model:backward(inputs, gradOutputs)
                    return err, gradParameters[ind]
                end 
            end 
        end
        
        if opt.freezeWeights == 0 then 
            feval = function(x)
                model:zeroGradParameters()
                --print(input:size())
                outputs = model:forward(inputs)
                err = criterion:forward(outputs, labels)
                local gradOutputs = criterion:backward(outputs, labels)
                model:backward(inputs, gradOutputs)
                return err, gradParameters
            end
        optim.sgd(feval, parameters, optimState) 
        end
        
    elseif opt.optim == 'adam' then 
        optim.adam(feval, parameters, optimState)
    end    
   

   batchNumber = batchNumber + 1
   loss_epoch = loss_epoch + err
    print(('[%d][%d/%d]\tTime %.3f Err %.4f LR %.0e WD %.0e M %.0e'):format(
            epoch, batchNumber,epochSize, timer:time().real, err,
            optimState.learningRate,optimState.weightDecay,optimState.momentum))
    dataTimer:reset()
    return outputs
end



--********************************************************************
-- TESTING/VALIDATION 
--********************************************************************

function getTestValidation(value) -- LOAD VAL/TEST accordingly value = 1 for validation
    if value   == 0 then
        print('==> doing epoch on test data:')
        return TestData,TestLabels,testLogger,'test','TESTING',TestContinuous
    elseif value == 1 then
        print('==> doing epoch on validation data:')
        return ValData,ValLabels,valLogger,'val','VAL',ValContinuous
    end   
end
 
function test_both(ibtest)
    if opt.criterion == 'joint' then
        print("==> Reseting Euclidean Weights for Loss function")
        criterion.criterions[2].weights:copy(torch.Tensor(gClass_weights))
    elseif  opt.criterion == 'disc' then
        print("==> Reseting Euclidean Weights for Loss function")
        criterion.weights:copy(torch.Tensor(gClass_weights))
    end
    
    -- test/validation switch       
    local DataToTest,LabelsToTest_disc,tLogger,tFileName,tLOGName,LabelsToTest_cont = getTestValidation(ibtest)
    print("==> online epoch # " .. epoch)
    local nTest = #DataToTest
    local iNum_MultiPatch = 0
    
    iTestBatchSize = BATCH_SIZE
    
    ---------------------
    -- Memory allocation
    ---------------------
    print('==> allocating minibatch memory')
    if opt.model_type == 'B' then 
        x_body_test = torch.Tensor(iTestBatchSize, 3, IMGSizes.BODY_SIZE, IMGSizes.BODY_SIZE)       
        x_body_test = x_body_test:cuda()
    elseif opt.model_type == 'I' then
        x_img_test = torch.Tensor(iTestBatchSize, 3, IMGSizes.INPUT_LARGEST_SIZE, IMGSizes.INPUT_LARGEST_SIZE)
        x_img_test = x_img_test:cuda()
    elseif opt.model_type == 'BI' then 
        x_img_test = torch.Tensor(iTestBatchSize, 3, IMGSizes.INPUT_LARGEST_SIZE, IMGSizes.INPUT_LARGEST_SIZE)
        x_body_test = torch.Tensor(iTestBatchSize, 3, IMGSizes.BODY_SIZE, IMGSizes.BODY_SIZE)  
        x_img_test = x_img_test:cuda()
        x_body_test = x_body_test:cuda()     
    end

    
    print '==> allocating global-test memory'
    if opt.criterion == 'joint' then 
        yt_global_cont = torch.Tensor(#DataToTest, NUM_CLASSES_CONTINUOUS):fill(0)
        ytbin_global_cont = torch.Tensor(#DataToTest, NUM_CLASSES_CONTINUOUS):fill(0)
        yest_global_cont = torch.Tensor(#DataToTest, NUM_CLASSES_CONTINUOUS):fill(0)
        yt_test_cont = torch.Tensor(iTestBatchSize, NUM_CLASSES_CONTINUOUS):fill(0)

        yt_global_disc = torch.Tensor(#DataToTest, NUM_CLASSES):fill(0)
        ytbin_global_disc = torch.Tensor(#DataToTest, NUM_CLASSES):fill(0)
        yest_global_disc = torch.Tensor(#DataToTest, NUM_CLASSES):fill(0)
        yt_test_disc = torch.Tensor(iTestBatchSize, NUM_CLASSES):fill(0) 
        
    elseif opt.criterion == 'cont' then 
        yt_global_cont = torch.Tensor(#DataToTest, NUM_CLASSES_CONTINUOUS):fill(0)
        ytbin_global_cont = torch.Tensor(#DataToTest, NUM_CLASSES_CONTINUOUS):fill(0)
        yest_global_cont = torch.Tensor(#DataToTest, NUM_CLASSES_CONTINUOUS):fill(0)
        yt_test_cont = torch.Tensor(iTestBatchSize, NUM_CLASSES_CONTINUOUS):fill(0)
        
    elseif opt.criterion == 'disc' then 
        yt_global_disc = torch.Tensor(#DataToTest, NUM_CLASSES):fill(0)
        ytbin_global_disc = torch.Tensor(#DataToTest, NUM_CLASSES):fill(0)
        yest_global_disc = torch.Tensor(#DataToTest, NUM_CLASSES):fill(0)
        yt_test_disc = torch.Tensor(iTestBatchSize, NUM_CLASSES):fill(0)
    end

    batchNumber = 0   
    timer:reset()   
    
    ------------------------
    -- Model Evaluation Mode
    ------------------------
    model:evaluate()
    test_loss = 0
    local total_test = 0
    local N = 0
    
    for t = 1, nTest, iTestBatchSize do
        if (t + iTestBatchSize - 1) > nTest then
            break
        end
        
        local idx = 1
        for indAnn = t,(t+iTestBatchSize-1) do                  
            filename = BASE_IMAGE_FOLDER .. '/' .. DataToTest[indAnn].folder .. '/' ..  DataToTest[indAnn].filename
            local ImgPatches = GetImagePatches(filename,DataToTest[indAnn],IMGSizes)
            
            --Inputs
            if opt.model_type == 'B' then 
                x_body_test[idx] = ImgPatches[2]
            elseif opt.model_type == 'I' then
                x_img_test[idx] = ImgPatches[1] 
            elseif opt.model_type == 'BI' then 
                x_img_test[idx] = ImgPatches[1] 
                x_body_test[idx] = ImgPatches[2]
            end

            if opt.criterion == 'joint' then 
                -- Labels
                yt_test_cont[idx]:copy(LabelsToTest_cont[indAnn])
                yt_global_cont[t+idx-1]:copy(yt_test_cont[idx])
                ytbin_global_cont[t+idx-1]:copy(LabelsToTest_cont[indAnn]:gt(0))

                yt_test_disc[idx]:copy(LabelsToTest_disc[indAnn])
                yt_global_disc[t+idx-1]:copy(yt_test_disc[idx])
                ytbin_global_disc[t+idx-1]:copy(LabelsToTest_disc[indAnn]:gt(0))
                
            elseif opt.criterion == 'cont' then 
                -- Labels
                yt_test_cont[idx]:copy(LabelsToTest_cont[indAnn])
                yt_global_cont[t+idx-1]:copy(yt_test_cont[idx])
                ytbin_global_cont[t+idx-1]:copy(LabelsToTest_cont[indAnn]:gt(0))

            elseif opt.criterion == 'disc' then 
                --Labels
                yt_test_disc[idx]:copy(LabelsToTest_disc[indAnn])
                yt_global_disc[t+idx-1]:copy(yt_test_disc[idx])
                ytbin_global_disc[t+idx-1]:copy(LabelsToTest_disc[indAnn]:gt(0))
            end        
            idx = idx + 1           
        end

        -------------------
        -- Predictions
        -------------------
        
        local preds={}
        if opt.criterion == 'joint' then 
            if opt.model_type == 'BI' then 
                preds = testBatch({x_img_test,x_body_test}, {yt_test_cont:cuda(),yt_test_disc:cuda()})
            elseif opt.model_type == 'B' then 
                preds = testBatch(x_body_test, {yt_test_cont:cuda(),yt_test_disc:cuda()})
            elseif opt.model_type == 'I' then 
                preds = testBatch(x_img_test, {yt_test_cont:cuda(),yt_test_disc:cuda()}) 
            end

        elseif opt.criterion == 'cont' then
            if opt.model_type == 'BI' then 
                preds = testBatch({x_img_test,x_body_test}, yt_test_cont:cuda())
            elseif opt.model_type == 'B' then 
                preds = testBatch(x_body_test, yt_test_cont:cuda())
            elseif opt.model_type == 'I' then 
                preds = testBatch(x_img_test, yt_test_cont:cuda()) 
            end
        elseif opt.criterion == 'disc' then
            if opt.model_type == 'BI' then 
                preds = testBatch({x_img_test,x_body_test}, yt_test_disc:cuda())
            elseif opt.model_type == 'B' then 
                preds = testBatch(x_body_test, yt_test_disc:cuda())
            elseif opt.model_type == 'I' then 
                preds = testBatch(x_img_test, yt_test_disc:cuda()) 
            end
        end
            
        -- Store predictions
        local idx2 = 1
        for indAnn = t,t+iTestBatchSize-1 do          
            if opt.criterion == 'joint' then 
                yest_global_cont[t+idx2-1]:copy(preds[1][idx2])
                yest_global_disc[t+idx2-1]:copy(preds[2][idx2])
            elseif opt.criterion == 'cont' then
                yest_global_cont[t+idx2-1]:copy(preds[idx2])
            elseif opt.criterion == 'disc' then
                yest_global_disc[t+idx2-1]:copy(preds[idx2])
            end
            idx2= idx2 + 1
        end        
        N = N + 1
        total_test = total_test + iTestBatchSize
        batchNumber = batchNumber + iTestBatchSize
    end 
  
    if opt.criterion == 'joint' then 
        --------------------------------------------------------
        --Evaluation here, we use the binary ones and the output
        --------------------------------------------------------
        print("==> " .. tLOGName .. "-CURVE EVALUATION ")
        auc_avg,gauc,ap_avg,gftap = CurveEvaluation(yest_global_disc, LabelsToTest_disc,gClass_naming)
        print("==> Averaged error Continuous: ==")
        cont_global_error = torch.sqrt(torch.mean(torch.cmul(yest_global_cont-yt_global_cont,yest_global_cont-yt_global_cont),1))

        for indCont=1,#Nconcepts do
            print(string.format("MSE-%s:\t%.3f",Nconcepts[indCont],cont_global_error[1][indCont]))
        end  
        print("==================")

        local test_loss_batch = test_loss / N --per batch loss
        print(string.format('[%d][' .. tLOGName .. ' SUMMARY] Total Time(s): %.2f \t' .. 'loss/batch: %.2f \n ' .. 'avg. AUC: %.2f \t ' .. 'glb avg. AUC: %.2f \t ' .. 'avg-AP: %.2f \t ',epoch, timer:time().real, test_loss_batch,auc_avg,gauc,ap_avg))
        print('\n')

        --------------------------------------
        -- SAVE logs, predictoins, files here
        --------------------------------------
        tLogger:add{epoch,test_loss_batch,auc_avg,gauc,ap_avg,cont_global_error[1][1],cont_global_error[1][2],cont_global_error[1][3]}

        local matfilename=paths.concat(opt.save, tFileName .. '_predictions_disc_' .. epoch .. '.mat')
        mat.save(matfilename , yest_global_disc:double())
        matfilename=paths.concat(opt.save, tFileName .. '_gt_' .. epoch .. '.mat')
        mat.save(matfilename , yt_global_disc:double())
        matfilename=paths.concat(opt.save, tFileName  .. '_gtBin_' .. epoch .. '.mat')
        mat.save(matfilename , ytbin_global_disc:double())
        matfilename=paths.concat(opt.save, tFileName  .. '_predictions_cont_' .. epoch .. '.mat')
        mat.save(matfilename , yest_global_cont:double())
        matfilename=paths.concat(opt.save, tFileName  .. '_gt_cont_' .. epoch .. '.mat')
        mat.save(matfilename , yt_global_cont:double()) 
    elseif opt.criterion == 'disc' then

        --------------------------------------------------------
        --Evaluation here, we use the binary ones and the output
        --------------------------------------------------------
        print("==> " .. tLOGName .. "-CURVE EVALUATION ")
        auc_avg,gauc,ap_avg,gftap = CurveEvaluation(yest_global_disc, LabelsToTest_disc,gClass_naming)
        print("==> Averaged error Continuous: ==")
        local test_loss_batch = test_loss / N --per batch loss
        print(string.format('[%d][' .. tLOGName .. ' SUMMARY] Total Time(s): %.2f \t' .. 'loss/batch: %.2f \n ' .. 'avg. AUC: %.2f \t ' .. 'glb avg. AUC: %.2f \t ' .. 'avg-AP: %.2f \t ',epoch, timer:time().real, test_loss_batch,auc_avg,gauc,ap_avg))
        print('\n')
        
        --------------------------------------
        -- SAVE logs, predictoins, files here
        --------------------------------------
        tLogger:add{epoch,test_loss_batch,auc_avg,gauc,ap_avg,0,0,0}

        matfilename=paths.concat(opt.save, tFileName .. '_predictions_disc_' .. epoch .. '.mat')
        mat.save(matfilename , yest_global_disc:double())
        matfilename=paths.concat(opt.save, tFileName .. '_gt_' .. epoch .. '.mat')
        mat.save(matfilename , yt_global_disc:double())
        matfilename=paths.concat(opt.save, tFileName  .. '_gtBin_' .. epoch .. '.mat')
        mat.save(matfilename , ytbin_global_disc:double())
        
    elseif opt.criterion == 'cont' then 
        --------------------------------------------------------
        --Evaluation here, we use the binary ones and the output
        --------------------------------------------------------
        print("==> Averaged error Continuous: ==")
        cont_global_error = torch.sqrt(torch.mean(torch.cmul(yest_global_cont-yt_global_cont,yest_global_cont-yt_global_cont),1))

        for indCont=1,#Nconcepts do
            print(string.format("MSE-%s:\t%.3f",Nconcepts[indCont],cont_global_error[1][indCont]))
        end  
        print("==================")

        local test_loss_batch = test_loss / N --per batch loss
        
        --------------------------------------
        -- SAVE logs, predictoins, files here
        --------------------------------------
        tLogger:add{epoch,test_loss_batch,0,0,0,cont_global_error[1][1],cont_global_error[1][2],cont_global_error[1][3]}
        matfilename=paths.concat(opt.save, tFileName  .. '_predictions_cont_' .. epoch .. '.mat')
        mat.save(matfilename , yest_global_cont:double())
        matfilename=paths.concat(opt.save, tFileName  .. '_gt_cont_' .. epoch .. '.mat')
        mat.save(matfilename , yt_global_cont:double()) 
    end
        
    collectgarbage()
end


function testBatch(inputData, labelsData)
    collectgarbage()
    local outputs = model:forward(inputData)
    local err = criterion:forward(outputs, labelsData)
    local pred = outputs
    test_loss = test_loss + err
    
    if batchNumber % (100*iTestBatchSize) == 0 then     
        print(('[%d][%d/%d] -- %.4f'):format(epoch, batchNumber, nTest,err))
    end
    return pred,err
end