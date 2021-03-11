--******************************************************************************************************
--  Combined B+I model
--******************************************************************************************************
function CreateEmotionModel_BI(InGPU,IMGSizes)
    require 'cudnn'
    require 'cunn'
    require 'optim'
    
    local nGPU=InGPU or 1;    
    local NUM_CLASSES = IMGSizes.NUM_CLASSES or 26;--we know this
    local NUM_CONCEPTS = IMGSizes.NUM_CLASSES_CONTINUOUS or 3;--we know this

    --[[ Building Image Model ]]--
    local imageModel = {}
    if opt.preTrainedImage == 'none' then
        print("nopretrained, building network from scratch for Image model")
        --novel model based on alexnet
        imageModel = createEmotionModelNeo()
        --return
    else
        local image_model_filename = paths.concat(opt.preTrainedFolder,opt.preTrainedImage)
        imageModel = createImageModel_pretrained(nGPU,IMGSizes,image_model_filename):cuda()
        print(" Preloaded Image Model: " .. opt.preTrainedImage)
    end
  
    -- [[ Building Body Model ]] --
    local bodyModel = {}
    if opt.preTrainedBodyOption == 'Alexnet' then 
        local model_filename = paths.concat(opt.preTrainedFolder,opt.preTrainedBodyAlexnet)
        bodyModel = createBodyModel_pretrained_alexnet(nGPU,IMGSizes,model_filename):cuda()
        print(" Preloaded Body Model based on Alexnet: " .. opt.preTrainedBodyAlexnet)
    elseif opt.preTrainedBodyOption == 'Decomposeme' then 
        local model_filename = paths.concat(opt.preTrainedFolder,opt.preTrainedBodyDecomposeme)
        bodyModel = createBodyModel_pretrained_decomposeme(nGPU,IMGSizes,model_filename):cuda()
        print(" Preloaded Body Model based on Decomposeme: " .. opt.preTrainedBodyDecomposeme)
    elseif opt.preTrainedBodyOption == 'SHG' then 
        local model_filename = paths.concat(opt.preTrainedFolder,opt.preTrainedBodySHG)
        bodyModel = createBodyModel_pretrained_shg(nGPU,IMGSizes,model_filename):cuda()
        print(" Preloaded Body Model based on SHG: " .. opt.preTrainedBodySHG)
    elseif  opt.preTrainedBodyOption == 'scratch' then 
        print('No pretrained, structuring from scratch for Body model')
        -- novel model based on alexnet 
        bodyModel = createEmotionModelNeo()
        --return
    end
    
    local NUM_FEATS_IMAGE = 0
    local l_img = imageModel:cuda():forward(torch.Tensor(BATCH_SIZE,3,IMGSizes.INPUT_LARGEST_SIZE,IMGSizes.INPUT_LARGEST_SIZE):cuda())
    NUM_FEATS_IMAGE = l_img:size(2) 
    --collectgarbage()
    
    local NUM_FEATS_BODY=0
    local l_body = bodyModel:cuda():forward(torch.Tensor(BATCH_SIZE,3,IMGSizes.BODY_SIZE,IMGSizes.BODY_SIZE):cuda())
    NUM_FEATS_BODY = l_body:size(2)
      
    --Total features
    local NUM_FEATS = NUM_FEATS_IMAGE + NUM_FEATS_BODY;
    
    local parallel= nn.ParallelTable()
    parallel:add(imageModel)
    parallel:add(bodyModel);
    
    -------------------
    -- Model definition
    -------------------
    local model=nn.Sequential()
    model:add(parallel)
    model:add(nn.JoinTable(2))
    local DROP_FIRST_CLASS=256

    local continuous_class = {}
    local categories_class = {}
    local mlp = {}
    if opt.criterion == 'joint' then
        continuous_class = nn.Linear(DROP_FIRST_CLASS, NUM_CONCEPTS)
        continuous_class = winit_layer(continuous_class, 'xavier')
        categories_class = nn.Linear(DROP_FIRST_CLASS, NUM_CLASSES)
        categories_class = winit_layer(categories_class,'xavier')
        mlp = nn.ConcatTable()
        mlp:add(continuous_class)
        mlp:add(categories_class)
    elseif opt.criterion == 'disc' then
        categories_class = nn.Linear(DROP_FIRST_CLASS, NUM_CLASSES)
        categories_class = winit_layer(categories_class,'xavier')
        mlp = categories_class
    elseif opt.criterion == 'cont' then
        continuous_class = nn.Linear(DROP_FIRST_CLASS, NUM_CONCEPTS)
        continuous_class = winit_layer(continuous_class, 'xavier')
        mlp = continuous_class
    end
    
    local class=nn.Sequential()
    local fclast=nn.Linear(NUM_FEATS, DROP_FIRST_CLASS)
    fclast = winit_layer(fclast, 'xavier')
    class:add(fclast)
    --class:add(nn.BatchNormalization(DROP_FIRST_CLASS,1e-3))
    --class:add(nn.ReLU(true))
    
    --Dropout
    if opt.doDrop > 0 then
        class:add(nn.Dropout(opt.doDrop))
    end
    
    class:add(mlp)
    model:add(class)
    model:cuda()
    
    return model 

    -------------------------------------------------    
    --[[ parallelize the model with Data Parallelism
    -------------------------------------------------
    gpus = torch.range(1, cutorch.getDeviceCount()):totable()
    dpt = nn.DataParallelTable(1):add(model, gpus):cuda()
    print(dpt) 
    
    return dpt
    --[[local ref_model = model 
    local parallelContainer = nn.DataParallel(1)
    parallelContainer:add(ref_model)
    print('Number of available GPUs', InGPU)
    for i=2,InGPU do 
        parallelContainer:add(ref_model:clone())
    end    
    model = parallelContainer
    print(model)
    return model
    ]]--
end

--******************************************************************************************************
-- Only Body Model
--******************************************************************************************************
function CreateEmotionModel_B(InGPU,IMGSizes)
    require 'cudnn'
    require 'cunn'
    local NUM_CLASSES = IMGSizes.NUM_CLASSES or 26;--we know this
    local NUM_CONCEPTS = IMGSizes.NUM_CLASSES_CONTINUOUS or 3;--we know this

    -- Building Body Model
    local bodyModel = {}
    if opt.preTrainedBodyOption == 'Alexnet' then 
        local model_filename = paths.concat(opt.preTrainedFolder,opt.preTrainedBodyAlexnet)
        bodyModel = createBodyModel_pretrained_alexnet(nGPU,IMGSizes,model_filename):cuda()
        print(" Preloaded Body Model based on Alexnet: " .. opt.preTrainedBodyAlexnet)
    elseif opt.preTrainedBodyOption == 'Decomposeme' then 
        local model_filename = paths.concat(opt.preTrainedFolder,opt.preTrainedBodyDecomposeme)
        bodyModel = createBodyModel_pretrained_decomposeme(nGPU,IMGSizes,model_filename):cuda()
        print(" Preloaded Body Model based on Decomposeme: " .. opt.preTrainedBodyDecomposeme)
    elseif opt.preTrainedBodyOption == 'SHG' then 
        local model_filename = paths.concat(opt.preTrainedFolder,opt.preTrainedBodySHG)
        bodyModel = createBodyModel_pretrained_shg(nGPU,IMGSizes,model_filename):cuda()
        print(" Preloaded Body Model based on SHG: " .. opt.preTrainedBodySHG)
    else 
        print('No pretrained, structuring from scratch for Body model')
        -- Insert the novel model here. 
        return
    end
    

    local NUM_FEATS_BODY=0
    local l_body = bodyModel:cuda():forward(torch.Tensor(BATCH_SIZE,3,IMGSizes.BODY_SIZE,IMGSizes.BODY_SIZE):cuda())
    NUM_FEATS_BODY = l_body:size(2)
    
    --Total Features
    local NUM_FEATS = NUM_FEATS_BODY
    local DROP_FIRST_CLASS = 256
    
    -- Model Definition
    local model=nn.Sequential()
    model:add(bodyModel)
     
    local continuous_class = {}
    local categories_class = {}
    local classifier = {}
    if opt.criterion == 'joint' then
        continuous_class = nn.Linear(DROP_FIRST_CLASS, NUM_CONCEPTS)
        categories_class = nn.Linear(DROP_FIRST_CLASS, NUM_CLASSES)
        categories_class = winit_layer(categories_class,'xavier')
        continuous_class = winit_layer(continuous_class, 'xavier')
        classifier = nn.ConcatTable()
        classifier:add(continuous_class)
        classifier:add(categories_class)
    elseif opt.criterion == 'disc' then
        categories_class = nn.Linear(DROP_FIRST_CLASS, NUM_CLASSES)
        categories_class = winit_layer(categories_class,'xavier')
        classifier = categories_class
    elseif opt.criterion == 'cont' or opt.criterion == 'cont_val' then
        continuous_class = nn.Linear(DROP_FIRST_CLASS, NUM_CONCEPTS)
        continuous_class = winit_layer(continuous_class, 'xavier')
        classifier = continuous_class
    end
    
    local fusion_fc=nn.Sequential()
    local fclast=nn.Linear(NUM_FEATS, DROP_FIRST_CLASS)
    fclast = winit_layer(fclast, 'xavier')
    fusion_fc:add(fclast)
    --class:add(nn.BatchNormalization(DROP_FIRST_CLASS,1e-3)) --not required as the last layer is FC
    --class:add(nn.ReLU(true))  --not required as the last layer is FC
    
    --Dropout
    if opt.doDrop > 0 then
        fusion_fc:add(nn.Dropout(opt.doDrop))
    end
    
    fusion_fc:add(classifier)
    model:add(fusion_fc)
    model:cuda()
    return model
    
end

--******************************************************************************************************
--  Only Image Model
--******************************************************************************************************
function CreateEmotionModel_I(InGPU,IMGSizes)
    require 'cudnn'
    require 'cunn'
    local NUM_CLASSES = IMGSizes.NUM_CLASSES or 26;--we know this
    local NUM_CONCEPTS = IMGSizes.NUM_CLASSES_CONTINUOUS or 3;--we know this
    
    -- Building Image Model 
    local imageModel = {}
    if opt.preTrainedImage == 'none' then
        print("building the network from scratch for Image model")
        -- Insert the novel model here. 
        return
    else
        local image_model_filename = paths.concat(opt.preTrainedFolder,opt.preTrainedImage)
        imageModel = createImageModel_pretrained(nGPU,IMGSizes,image_model_filename):cuda()
        print(" Preloaded Image Model: " .. opt.preTrainedImage)
    end
    
    local NUM_FEATS_IMAGE = 0
    local l_img = imageModel:cuda():forward(torch.Tensor(BATCH_SIZE,3,IMGSizes.INPUT_LARGEST_SIZE,IMGSizes.INPUT_LARGEST_SIZE):cuda())
    NUM_FEATS_IMAGE = l_img:size(2)
    
    -- Total Features
    local NUM_FEATS = NUM_FEATS_IMAGE
    local DROP_FIRST_CLASS = 256

    -- Model Definition
    local model=nn.Sequential()
    model:add(imageModel)

    local continuous_class = {}
    local categories_class = {}
    local classifier = {}
    if opt.criterion == 'joint' then
        continuous_class = nn.Linear(DROP_FIRST_CLASS, NUM_CONCEPTS)
        continuous_class = winit_layer(continuous_class, 'xavier')        
        categories_class = nn.Linear(DROP_FIRST_CLASS, NUM_CLASSES)
        categories_class = winit_layer(categories_class,'xavier')
        classifier = nn.ConcatTable()
        classifier:add(continuous_class)
        classifier:add(categories_class)
    elseif opt.criterion == 'disc' then
        categories_class = nn.Linear(DROP_FIRST_CLASS, NUM_CLASSES)
        categories_class = winit_layer(categories_class,'xavier')
        classifier = categories_class
    elseif opt.criterion == 'cont' or opt.criterion == 'cont_val' then
        continuous_class = nn.Linear(DROP_FIRST_CLASS, NUM_CONCEPTS)
        continuous_class = winit_layer(continuous_class, 'xavier')
        classifier = continuous_class
    end
    
    local fusion_fc = nn.Sequential()
    local fclast = nn.Linear(NUM_FEATS, DROP_FIRST_CLASS)
    fclast = winit_layer(fclast, 'xavier')
    fusion_fc:add(fclast)
    --fusion_fc:add(nn.BatchNormalization(DROP_FIRST_CLASS,1e-3))
    --fusion_fc:add(nn.ReLU(true))
    
    --Dropout
    if opt.doDrop > 0 then
        fusion_fc:add(nn.Dropout(opt.doDrop))
    end
    
    fusion_fc:add(classifier)
    model:add(fusion_fc)
    model:cuda()
    return model
end   

--******************************************************************************************************
--  Combined B+I model - ONLY for Valence
--******************************************************************************************************
function CreateEmotionModel_BI_valence(InGPU,IMGSizes)
    require 'cudnn'
    require 'cunn'
    local nGPU=InGPU or 1;    
    local NUM_CONCEPTS = IMGSizes.NUM_CLASSES_CONTINUOUS -1;

    -- Building Image Model 
    imageModel = {}
    if opt.preTrainedImage == nil then
        print("nopretrained, structuring from scratch for Image model")
        -- Insert the novel model here. 
        return
    else
        local image_model_filename = paths.concat(opt.preTrainedFolder,opt.preTrainedImage)
        imageModel = createImageModel_pretrained(nGPU,IMGSizes,image_model_filename):cuda()
        print(" Preloaded Image Model: " .. opt.preTrainedImage)
    end
  
    -- Building Body Model
    local bodyModel = {}
    if opt.preTrainedBodyOption == 'Alexnet' then 
        local model_filename = paths.concat(opt.preTrainedFolder,opt.preTrainedBodyAlexnet)
        bodyModel = createBodyModel_pretrained_alexnet(nGPU,IMGSizes,model_filename):cuda()
        print(" Preloaded Body Model based on Alexnet: " .. opt.preTrainedBodyAlexnet)
    elseif opt.preTrainedBodyOption == 'Decomposeme' then 
        local model_filename = paths.concat(opt.preTrainedFolder,opt.preTrainedBodyDecomposeme)
        bodyModel = createBodyModel_pretrained_decomposeme(nGPU,IMGSizes,model_filename):cuda()
        print(" Preloaded Body Model based on Decomposeme: " .. opt.preTrainedBodyDecomposeme)
    elseif opt.preTrainedBodyOption == 'SHG' then 
        local model_filename = paths.concat(opt.preTrainedFolder,opt.preTrainedBodySHG)
        bodyModel = createBodyModel_pretrained_shg(nGPU,IMGSizes,model_filename):cuda()
        print(" Preloaded Body Model based on SHG: " .. opt.preTrainedBodySHG)
    else 
        print('No pretrained, structuring from scratch for Body model')
        -- Insert the novel model here. 
        return
    end
    
    --uncomment for other models except SHG
    local NUM_FEATS_BODY=0
    local l_body = bodyModel:cuda():forward(torch.Tensor(2,3,IMGSizes.BODY_SIZE,IMGSizes.BODY_SIZE):cuda())
    NUM_FEATS_BODY =  l_body:size(2) --1024 for SHG
    --print(NUM_FEATS_BODY)
  
    local NUM_FEATS_IMAGE=0
    local l_img = imageModel:cuda():forward(torch.Tensor(2,3,IMGSizes.INPUT_LARGEST_SIZE,IMGSizes.INPUT_LARGEST_SIZE):cuda())
    NUM_FEATS_IMAGE = l_img:size(2)
    
    --Total features
    local NUM_FEATS=NUM_FEATS_IMAGE+NUM_FEATS_BODY
    
    local parallel= nn.ParallelTable()
    parallel:add(imageModel)
    parallel:add(bodyModel);
    
    -------------------
    -- Model definition
    -------------------
    local model=nn.Sequential()
    model:add(parallel)
    model:add(nn.JoinTable(2))
    local DROP_FIRST_CLASS=256

    local continuous_class = {}
    local categories_class = {}
    local mlp = {}
    if opt.criterion == 'joint' then
        continuous_class = nn.Linear(DROP_FIRST_CLASS, NUM_CONCEPTS)
        continuous_class = winit_layer(continuous_class, 'xavier')
        categories_class = nn.Linear(DROP_FIRST_CLASS, NUM_CLASSES)
        categories_class = winit_layer(categories_class,'xavier')
        mlp = nn.ConcatTable()
        mlp:add(continuous_class)
        mlp:add(categories_class)
    elseif opt.criterion == 'disc' then
        categories_class = nn.Linear(DROP_FIRST_CLASS, NUM_CLASSES)
        categories_class = winit_layer(categories_class,'xavier')
        mlp = categories_class
    elseif opt.criterion == 'cont' or opt.criterion == 'cont_val' then
        continuous_class = nn.Linear(DROP_FIRST_CLASS, NUM_CONCEPTS)
        continuous_class = winit_layer(continuous_class, 'xavier')
        mlp = continuous_class
    end
    
    local class=nn.Sequential()
    local fclast=nn.Linear(NUM_FEATS, DROP_FIRST_CLASS)
    fclast = winit_layer(fclast, 'xavier')
    class:add(fclast)
    class:add(nn.BatchNormalization(DROP_FIRST_CLASS,1e-3))
    class:add(nn.ReLU(true))
    
    --Dropout
    if opt.doDrop > 0 then
        class:add(nn.Dropout(opt.doDrop))
    end
    
    class:add(mlp)
    model:add(class)
    model:add(nn.Sigmoid())
    model:cuda()
    return model
end


--******************************************************************************************************
-- Pretrained Model - Image, based on PlacesCNN and DeComposemMe 
--******************************************************************************************************
function createImageModel_pretrained(nGPU,IMGSizes,model_filename)
    require 'cudnn'   
    
    model_filename = model_filename -- or 'give/absolutePath'
    local model_pretrained = torch.load(model_filename)
    local features = model_pretrained:get(1)
    
    if torch.type(features) == 'nn.DataParallel' then
        features = features:get(1)
        features:evaluate()
    end
    
    --local model = nn.Sequential()
    --model:add(features)
    --model:add(nn.View(k:size(2)))
     
    --[[ Freezing weights of the pretrained network 
    if opt.freezeWeights == 1 then 
        features:apply(function(m) 
                m.updateGradInput = function(self, inp, out) end
                m.accGradParameters = function(self,inp, out) end
            end)
    end
    ]]--
    
    --[[ Freezing weights of the pretrained network 
    if opt.freezeWeights == 1 then 
        features:apply(function(m) 
                if torch.typename(m) == 'cudnn.SpatialConvolution' then 
                    m.updateGradInput = function(self, inp, out) end
                    m.accGradParameters = function(self,inp, out) end
                end
            end)
    end
    ]]--
    
    --[[ Freezing weights of the pretrained network ]]--
    if opt.freezeWeights == 1 then 
        features.updateGradInput = function(self, inp, out) end
        features.accGradParameters = function(self, inp, out) end
    end
    
    --[[ Replacing SpatialAvgPooling with MaxPooling ]]--
    if opt.image_last_layer == 'MP' then 
        features:replace(function(module)
                if torch.typename(module) == 'nn.SpatialAveragePooling' then
                    return cudnn.SpatialMaxPooling(4,4,1,1)
                else
                    return module
                end
            end)
    end
        
    local k=features:cuda():forward(torch.Tensor(2,3,224,224):cuda())
    
    return features:add(nn.View(k:size(2)))
    --[[ testing with scratch models
    net = torch.load(model_filename):unpack()
    net:add(cudnn.SpatialMaxPooling(3,3,2,2))
    net:add(nn.View(256*7*7))
    return net
    ]]--
    
    --[[ Resnets 
    resnet = torch.load(model_filename)
    --replace the FC layer with Identity
    resnet:replace(function(module)
            if torch.typename(module) == 'nn.Linear' then
                return nil
            else
                return module
            end
        end)
    --add bias, gradBias terms initialized with zeros 
    resnet:replace(function(module)
        if torch.typename(module) == 'cudnn.SpatialConvolution' then
            module.bias = torch.zeros(module.weight:size(1)):float()
            module.gradBias = torch.zeros(module.weight:size(1)):float()
            --module.gradBias = module.weight:size(1):float()
      return cudnn.convert(module)
   else
      return module
   end
end)

    return resnet
  ]]--  
    
    ----------------------------------
    --PlacesCNN (without BN layers)
    ----------------------------------
    --[[
    require 'loadcaffe'
    local path_pretrained = '/root/torch/cvpr/cvpr_Empathy/pretrained_models/places/'    
    full_model = loadcaffe.load(paths.concat(path_pretrained,'places205CNN_deploy_upgraded.prototxt'), paths.concat(path_pretrained,'places205CNN_iter_300000_upgraded.caffemodel'), 'cudnn')
    features = nn.Sequential()
    for ind=1,16 do 
        features:add(full_model.modules[ind])
    end
return features   
        ]]--
    
    --[[ DecomposeMe
    local model_pretrained = torch.load(model_filename)
    local features = model_pretrained:get(1)  
    if torch.type(features) == 'nn.DataParallel' then
        features = features:get(1);
    end
    features:replace(function(module)
            if torch.typename(module) == 'nn.SpatialBatchNormalization' then
                return cudnn.convert(module)
            else
                return module
            end
        end)    
    
    local k=features:cuda():forward(torch.Tensor(2,3,IMGSizes.INPUT_LARGEST_SIZE,IMGSizes.INPUT_LARGEST_SIZE):cuda());
    local model = nn.Sequential();
    model:add(features);   
    model:add(nn.View(k:size(2)));   
    return model;  
    
    ]]--
    
    -- Set all the values of running_mean/var and save_mean, var to normal distribution 
    --[[
    bn_nodes = features:findModules('nn.SpatialBatchNormalization')
    for ind = 1,#bn_nodes do 
        bn_nodes[ind].running_var = torch.randn(bn_nodes[ind].running_var:size())
        bn_nodes[ind].running_mean = torch.randn(bn_nodes[ind].running_mean:size())
        bn_nodes[ind].save_mean = torch.randn(bn_nodes[ind].save_mean:size())
        bn_nodes[ind].save_std = torch.randn(bn_nodes[ind].save_std:size())
    end
    ]]--
    
    --[[
    -- OR set all the values of running_mean/var and save_mean/var to 0 and 1 respectively.
    bn_nodes = features:findModules('nn.SpatialBatchNormalization')
    for ind = 1,#bn_nodes do 
        bn_nodes[ind].running_var  = torch.CudaTensor(bn_nodes[ind].running_var:size()):fill(1)
        bn_nodes[ind].running_mean = torch.CudaTensor(bn_nodes[ind].running_mean:size()):fill(0)
        bn_nodes[ind].save_mean    = torch.CudaTensor(bn_nodes[ind].save_mean:size()):fill(0)
        bn_nodes[ind].save_std     = torch.CudaTesnor(bn_nodes[ind].save_std:size()):fill(1)
    end
    ]]--
    
    -- Replace nn.BN with cudnn.BN 

end

--******************************************************************************************************
-- Pretrained Model - Body, based on DeComposemMe 
--******************************************************************************************************
function createBodyModel_pretrained_decomposeme(nGPU,IMGSizes,model_filename)
    require 'cudnn'       
    local features=torch.load(model_filename);
    
    --Choose between spatial average pooling or max pooling for the last layer here 
    features:add(nn.SpatialAveragePooling(4,4,1,1))
    
    local k=features:cuda():forward(torch.Tensor(2,3,IMGSizes.BODY_SIZE,IMGSizes.BODY_SIZE):cuda())
    features:evaluate()
    --setting the 1st BN layer values running_mean, running_var() as the mean and 
    -- var of all the other layers
    --[[
    bn_nodes = features:findModules('nn.SpatialBatchNormalization')
    local temp_mean = torch.Tensor(#bn_nodes)
    local temp_var = torch.Tensor(#bn_nodes)
    for ind = 1,#bn_nodes do 
        temp_mean[ind] = torch.sum(bn_nodes[ind].running_mean)
        temp_var[ind] = torch.sum(bn_nodes[ind].running_var)
        --print(torch.sum(bn_nodes[ind].running_mean))
    end
    bn_nodes[1].running_mean:fill(torch.mean(temp_mean[{{2,8}}]))
    bn_nodes[1].running_var:fill(torch.mean(temp_var[{{2,8}}]))
    ]]--
    
    -- OR set all the values of running_mean, var and save_mean, var to normal distribution 
    --[[
    bn_nodes = features:findModules('nn.SpatialBatchNormalization')
    for ind = 1,#bn_nodes do 
        bn_nodes[ind].running_var = torch.randn(bn_nodes[ind].running_var:size())
        bn_nodes[ind].running_mean = torch.randn(bn_nodes[ind].running_mean:size())
        bn_nodes[ind].save_mean = torch.randn(bn_nodes[ind].save_mean:size())
        bn_nodes[ind].save_std = torch.randn(bn_nodes[ind].save_std:size())
    end        
      ]]--
  

    -- OR set all the values of running_mean/var and save_mean/var to 0 and 1 respectively.
    bn_nodes = features:findModules('nn.SpatialBatchNormalization')
    for ind = 1,#bn_nodes do 
        bn_nodes[ind].running_var  = torch.CudaTensor(bn_nodes[ind].running_var:size()):fill(1)
        bn_nodes[ind].running_mean = torch.CudaTensor(bn_nodes[ind].running_mean:size()):fill(0)
        bn_nodes[ind].save_mean    = torch.CudaTensor(bn_nodes[ind].save_mean:size()):fill(0)
        bn_nodes[ind].save_std     = torch.CudaTensor(bn_nodes[ind].save_std:size()):fill(1)
    end
    
    local k=features:cuda():forward(torch.Tensor(2,3,IMGSizes.BODY_SIZE,IMGSizes.BODY_SIZE):cuda())
    local model = nn.Sequential()
    model:add(features)
    model:add(nn.View(k:size(2)*k:size(3)*k:size(4)))
    model:cuda()
    return features:add(nn.View(k:size(2)))
end

--******************************************************************************************************
-- Pretrained Model - Body, based on Alexnet 
--******************************************************************************************************
function createBodyModel_pretrained_alexnet(nGPU,IMGSizes,model_filename)
       
    model_filename = model_filename -- or 'give/absolutePath'
    local features = torch.load(model_filename):unpack()
    
    --[[ Freezing weights of the pretrained network 
    if opt.freezeWeights == 1 then 
        features:apply(function(m) 
                m.updateGradInput = function(self, inp, out) end
                m.accGradParameters = function(self,inp, out) end
            end)
    end
    ]]--
    
    --[[ Freezing weights of the pretrained network 
    if opt.freezeWeights == 1 then 
        features:apply(function(m) 
                if torch.typename(m) == 'cudnn.SpatialConvolution' then 
                    m.updateGradInput = function(self, inp, out) end
                    m.accGradParameters = function(self,inp, out) end
                end
            end)
    end
    ]]--
    
    --[[ Freezing weights of the pretrained network ]]--
    if opt.freezeWeights == 1 then 
        features.updateGradInput = function(self, inp, out) end
        features.accGradParameters = function(self, inp, out) end
    end
    
    --[[ Choose between spatial average pooling or max pooling for the last layer here ]]--
    if opt.body_last_layer == "SAP" then 
        features:add(cudnn.SpatialAveragePooling(3,3,16,16))        
    elseif opt.body_last_layer == "MP" then 
        features:add(cudnn.SpatialMaxPooling(3,3,16,16))
    end

        
    local k=features:cuda():forward(torch.Tensor(2,3,IMGSizes.BODY_SIZE,IMGSizes.BODY_SIZE):cuda())
    --local model = nn.Sequential()
    --model:add(features)
    --model:add(nn.View(k:size(2)*k:size(3)*k:size(4)))
    
    return features:add(nn.View(k:size(2)*k:size(3)*k:size(4)))
    --[[ net = torch.load(model_filename):unpack()
    net:add(cudnn.SpatialMaxPooling(3,3,2,2))
    net:add(nn.View(256*4*4))
    return net
    ]]--
end  

--******************************************************************************************************
-- Pretrained Model - Body, based on SHG 
--******************************************************************************************************
function createBodyModel_pretrained_shg(nGPU,IMGSizes,model_filename)
    posemodel = torch.load(model_filename)
    function addModule(network)
        local input = network 
        local net   = nn.Sequential()
        net:add(input)
        net:add(nn.CAddTable())
        net:add(nn.Mean(2))
        net:add(nn.View(32*32))
        --[[
        net:add(nn.CAddTable())
        net:add(nn.View(16,32,32))
        net:add(nn.Mean(2))
        net:add(nn.Mean())

        local joinT = nn.JoinTable(1) -- '1' is the dimension along which the table of tensors are added.
        local mean  = nn.Mean() --takes the mean over the first dimension
        local fc    = nn.Linear(64*64, 4096)
        net:add(input)
        net:add(joinT) --joins the table of tensors (8 of {1x16x64x64} into [8x16x64x64]), and outputs a single tensor.
        --net:add(nn.Copy(nil,nil,true))
        net:add(mean) -- [8x16x64x64]-->[16x64x64]
        net:add(mean) -- [16x64x64]-->[64x64]
        net:add(nn.View(32*32)) --when using body_size=128, use (1,64,64) for body_size=256
        ]]--
        return net:cuda()    
    end
    return addModule(posemodel)
end

--******************************************************************************************************
-- Model Design from Scratch (ALexnet)
--******************************************************************************************************
function createEmotionModelNeo()
    model= nn.Sequential()
    
----CONV 1
    model:add(cudnn.SpatialConvolution(3,96,11,11,4,4,2,2))
        model.modules[#model.modules].weight:normal(0, 0.01)
        model.modules[#model.modules].bias:fill(0)
    model:add(cudnn.ReLU())
    model:add(nn.SpatialCrossMapLRN(5, 0.0001, 0.75, 1))
    model:add(nn.SpatialMaxPooling(3,3,2,2))

----CONV 2
    model:add(cudnn.SpatialConvolution(96,256,5,5,1,1,2,2))
        model.modules[#model.modules].weight:normal(0, 0.01)
        model.modules[#model.modules].bias:fill(0.1)
    model:add(cudnn.ReLU())
    model:add(nn.SpatialCrossMapLRN(5, 0.0001, 0.75, 1))
    model:add(nn.SpatialMaxPooling(3,3,2,2)) 
    
----CONV 3
    model:add(cudnn.SpatialConvolution(256,384,3,3,1,1,1,1))  
        model.modules[#model.modules].weight:normal(0, 0.01)
        model.modules[#model.modules].bias:fill(0)
    model:add(cudnn.ReLU())

----CONV 4    
    model:add(cudnn.SpatialConvolution(384,384,3,3,1,1,1,1))  
        model.modules[#model.modules].weight:normal(0, 0.01)
        model.modules[#model.modules].bias:fill(0.1)
    model:add(cudnn.ReLU())

----CONV 5
    model:add(cudnn.SpatialConvolution(384,256,3,3,1,1,1,1)) 
        model.modules[#model.modules].weight:normal(0, 0.01)
        model.modules[#model.modules].bias:fill(0.1)
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(3,3,2,2))  
    model:add(nn.View(256*6*6))
    return model:cuda()
end


--******************************************************************************************************
-- Define Criterion 
--******************************************************************************************************
function DefineCriterion()
    local criterion = {}
    if opt.criterion == 'joint' then -- Parallel criterion 
        criterion = nn.ParallelCriterion() 
        print('=>Using Joint Criterions (weightMSE+marginMSE)')      
        local cont = nn.marginMSE(opt.mMargin/opt.contNorm,true,opt.mSaturation/opt.contNorm)  
        criterion:add(cont)
        local disc = nn.weightMSE(gClass_weights,true)
        criterion:add(disc)
        print('=>Global Weigths set to: ')
        print(torch.view(gClass_weights,1,NUM_CLASSES))
        criterion:cuda()
        print('=>Setting Criterion weights:')
        criterion.weights[1] = opt.Wcont
        criterion.weights[2] = opt.Wdisc
    elseif opt.criterion == 'disc' then -- Discrete criterion 
        print('=>Using Discrete Criterion only (weightMSE)')      
        criterion = nn.weightMSE(gClass_weights,true)
        print('=>Global Weigths set to: ')
        print(torch.view(gClass_weights,1,NUM_CLASSES))
        criterion:cuda()
    elseif opt.criterion == 'cont' then -- Continuous criterion
        print('=>Using Continuous Criterion only (marginMSE)')
        criterion =  nn.marginMSE(opt.mMargin/opt.contNorm,true,opt.mSaturation/opt.contNorm)  
        criterion:cuda()
    elseif opt.criterion == 'cont_val' then 
        print('=>Using Continuous Criterion - (BCE)')
        criterion =  nn.MSECriterion()  
        criterion:cuda()
    end
    return criterion
end

--******************************************************************************************************
-- Initialization functions
--******************************************************************************************************
function w_init_xavier(fan_in, fan_out)
   return math.sqrt(2/(fan_in + fan_out))
end
-- "Understanding the difficulty of training deep feedforward neural networks"
-- Xavier Glorot, 2010
function w_init_xavier_caffe(fan_in, fan_out)
   return math.sqrt(1/fan_in)
end
-- "Efficient backprop"
-- Yann Lecun, 1998
function w_init_heuristic(fan_in, fan_out)
   return math.sqrt(1/(3*fan_in))
end
function winit(net, arg)
   -- choose initialization method
   local method = nil
   if     arg == 'heuristic'    then method = w_init_heuristic;
   elseif arg == 'xavier'       then method = w_init_xavier;
   elseif arg == 'xavier_caffe' then method = w_init_xavier_caffe;
   elseif arg == 'kaiming'      then method = w_init_kaiming;
   else
      assert(false);
   end
   -- loop over all convolutional modules
   for i = 1, #net.modules do
      local m = net.modules[i]
      if m.__typename == 'nn.SpatialConvolution' then
         m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW));
      elseif m.__typename == 'nn.SpatialConvolutionMM' then
         m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW));
      elseif m.__typename == 'nn.LateralConvolution' then
         m:reset(method(m.nInputPlane*1*1, m.nOutputPlane*1*1));
      elseif m.__typename == 'nn.VerticalConvolution' then
         m:reset(method(1*m.kH*m.kW, 1*m.kH*m.kW));
      elseif m.__typename == 'nn.HorizontalConvolution' then
         m:reset(method(1*m.kH*m.kW, 1*m.kH*m.kW));
      elseif m.__typename == 'nn.Linear' then
         m:reset(method(m.weight:size(2), m.weight:size(1)));
      elseif m.__typename == 'nn.TemporalConvolution' then
         m:reset(method(m.weight:size(2), m.weight:size(1)));
      elseif m.__typename == 'cudnn.SpatialConvolution' then
         local val=0;
         val = method(m.weight:size(2), m.weight:size(1));
         m.weight:uniform(-val, val);
         m.bias:uniform(-val, val);         
      else
      end
      if m.bias then
         m.bias:zero();
      end
   end
   return net;
   end
function winit_layer(layer, arg)
   -- choose initialization method
   local method = nil;
   method = w_init_xavier;   
   -- loop over all convolutional modules   
      local m = layer;
      if m.__typename == 'nn.SpatialConvolution' then
         m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW));
      elseif m.__typename == 'nn.SpatialConvolutionMM' then
         m:reset(method(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW));
      elseif m.__typename == 'nn.LateralConvolution' then
         m:reset(method(m.nInputPlane*1*1, m.nOutputPlane*1*1));
      elseif m.__typename == 'nn.VerticalConvolution' then
         m:reset(method(1*m.kH*m.kW, 1*m.kH*m.kW));
      elseif m.__typename == 'nn.HorizontalConvolution' then
         m:reset(method(1*m.kH*m.kW, 1*m.kH*m.kW));
      elseif m.__typename == 'nn.Linear' then
         m:reset(method(m.weight:size(2), m.weight:size(1)));
      elseif m.__typename == 'nn.TemporalConvolution' then
         m:reset(method(m.weight:size(2), m.weight:size(1)));
      elseif m.__typename == 'cudnn.SpatialConvolution' then
         local val=0;
         val = method(m.weight:size(2), m.weight:size(1));
         m.weight:uniform(-val, val);
         m.bias:uniform(-val, val);         
      else
      end
      if m.bias then
         m.bias:zero();
      end   
   return layer;
   end
