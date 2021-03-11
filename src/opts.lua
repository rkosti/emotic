--[[
     Options file containing all the options set up for all the experiments in EMOTIC
]]--

local M = { }
function M.parse(arg)
   local datadir='./';
   local defaultDir = paths.concat(datadir, '')
    local cmd = torch.CmdLine()
    cmd:text()    
    cmd:text()
    cmd:text('Options:')
    
    cmd:option('-modelname',             'CreateEmotionModel_BI',   'Model do be used, BI, B or I')
    cmd:option('-criterion',                              'joint',   'other options are disc and cont')
    
    ------------------------------------------------------
    ---------- Optimization options ----------------------
    ------------------------------------------------------
    cmd:option('-LR',          0.01, 'learning rate; if set, overrides default LR/WD recipe') --good for transfer learning
    cmd:option('-nItersLR',     7, 'number of epochs before reducing the learning rate in a factor of 10')
    cmd:option('-nEpochs',      14,    'Number of total epochs to run')
    cmd:option('-epochSize',   100,    'Number of batches per epoch')--doesn't matter, is decided by (batchSize/dataSize)
    cmd:option('-epochNumber',   1,    'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',    26*2,    'mini-batch size (1 = pure stochastic)')
    cmd:option('-testbatchSize', 4,    'mini-batch size (1 = pure stochastic)')
    cmd:option('-optim',     'sgd', 'Optimization method (sgd,lbfgs,adagrad,adadelta, adam)')
    cmd:option('-doDrop',      0.5,   'set to 0 to remove dropout in the last FC')    
    cmd:option('-momentum',    0.9,  'momentum')
    cmd:option('-weightDecay',   5e-4, 'weight decay - 5e-4')
    cmd:option('-freezeWeights', 0, 'whether to freeze the lower layer weights or not')
    
    ------------------------------------------------------    
    ------------ Augmentation options --------------------
    ------------------------------------------------------
    cmd:option('-dataAugment',    0, '1 for augmenting, 0 for not')
    cmd:option('-HorizontalFlip', 1, '1 for flipping horizontally, 0 otherwise')
    cmd:option('-Rotation',       0, 'rotation in degrees')
    cmd:option('-RandomCrop',    0, 'torch.random(1,56) = 25% of the image size - padding value')
    cmd:option('-brightness',     0, 'brightness(0.4), part of ColorJitter')
    cmd:option('-contrast',       0, 'contrast(0.4), part of ColorJitter')
    cmd:option('-saturation',     0, 'saturation(0.4), part of ColorJitter')

    ------------------------------------------------------
    ------------ General options -------------------------
    ------------------------------------------------------
    cmd:option('-cache', paths.concat(datadir, 'rerun_output_files'),'subdirectory in which to save/log experiments')
    cmd:option('-images', paths.concat(datadir, 'dataset/emotic/'),'Home of Empathy dataset')
    cmd:option('-manualSeed',         2, 'Manually set RNG seed -- for repeatability')
    cmd:option('-data',            'annotations/','Home of Annotation files')
    cmd:option('-GPU',                1, 'Default preferred GPU')
    cmd:option('-nGPU',               1, 'Number of GPUs to use by default')    
    --cmd:option('-nDonkeys',           2, 'number of donkeys to initialize (data loading threads)')
    cmd:option('-MTL',           'both',   '')    
    cmd:option('-Wcont',              1/2,   'Weight given to the conitnuous loss')
    cmd:option('-Wdisc',            1/2,   'Weight given to the discrete loss')
    cmd:option('-mSaturation',     1500,   'Value for which the loss saturates (to avoid outliers -the loss). 1500 is large enough dont be used. This will be normalized using contNorm.')
    cmd:option('-mMargin',            1,   'margin = abs(pred - label), for which the loss back-propagated is 0')
    cmd:option('-contNorm',          10,   'Set to 10 to normalize continuos annotations between 0 and 1.')
    cmd:option('-reWeight',           0,   'Set to 1 to reweight based on each batch information. Otherwise it is set once at the beginning.')
    cmd:option('-discLabel',          0,   'Set to 0 as lowest value in the class. Set to -1 for forcing labels going from -1 to 1')
    cmd:option('-discScale',          1,   'Set to N to have values beween {discLabel*N,N}. Set to -1 for forcing labels going from -1 to 1')
    --cmd:option('-discNL',                                  'none',   'none is a naked layer, Tanh is using hyperbolic tangent (for values -1,1) or any other.')
    cmd:option('-preTrainedFolder',         paths.concat(datadir, 'pretrained_models'),   'default pretrained models folder')
    cmd:option('-preTrainedImage',          'model_myVDavg_640_Places.t7',   'pretrained image models:model_myVDavg_640_Places.t7, resnet/resnet-18.t7,places/resnet50_places365.t7, alexnet/alexnet_features.t7. nil - for training from scratch')
    cmd:option('-preTrainedBodyOption',                 'Decomposeme',   'Chose between Alexnet, SHG and Decomposeme based models, and set to nil for building model from scratch')
    cmd:option('-preTrainedBodyDecomposeme',  'myVD_ImgNet_66_old.t7',   'pretrained body model based on DecomposeMe')
    cmd:option('-preTrainedBodyAlexnet',    'alexnet_features_new.t7',   'pretrained body model based on alexnet')
    cmd:option('-preTrainedBodySHG', 'hg/umich-stacked-hourglass.t7',   'pretrained pose model based on SHG')
    
    --------------------------------------------------------------
    ---------- Weight normalization options ----------------------
    --------------------------------------------------------------
    cmd:option('-normFactor',1.2, 'Weight in the logarithm to normalize the weights of the histogram');
    cmd:option('-normWeights',1, 'Weight in the logarithm to normalize the weights of the histogram');
    
    -----------------------------------------------------------
    ---------- Model options ----------------------------------
    -----------------------------------------------------------
    cmd:option('-model_type', 'BI', 'either BI, B or I') 
    cmd:option('-body_last_layer', 'SAP', 'either SAP (Spatial Average Pooling) or MP (Max Pooling)')
    cmd:option('-image_last_layer', 'SAP', 'either SAP (Spatial Average Pooling) or MP (Max Pooling)')
    --cmd:option('-netType',     'dec', 'Options: alexnet | overfeat')
    --cmd:option('-retrain',     'none', 'provide path to model to retrain with')
    cmd:option('-optimState',  'none', 'provide path to an optimState to reload from')
    cmd:option('-append', '','append another sub_folder if want to save differently inside the main output folder for organization purposes')
    cmd:text()

    local opt = cmd:parse(arg or {})
    opt.save = paths.concat(opt.cache, cmd:string(opt.append, opt, {append=true,images=true,retrain=true,optimState=true, cache=true, data=true, testbatchSize=true,nDonkeys=true}))
    local dataname = os.date():gsub(' ',''):gsub('/',''):gsub(':','')
    opt.save = paths.concat(opt.save, '' .. dataname)
    local LOG_APPEND = cmd:string(opt.append, opt, {append=true,images=true,retrain=true, optimState=true, cache=true, data=true,testbatchSize=true,nDonkeys=true})
    LOG_APPEND = LOG_APPEND .. '_' .. os.date('%a'..'%d'..'%b'..'%y'..'%X'):gsub(':','')
    opt.append = LOG_APPEND
    return opt, LOG_APPEND
end

return M
