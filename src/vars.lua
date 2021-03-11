--------------------------------------------------------------------------------------
-- Setting up the common libraries & variables shared across different training modes
--------------------------------------------------------------------------------------
require 'torch'
require 'paths'
require 'xlua'
require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'cutorch'
require 'optim'
require 'image'
tnt = require 'torchnet'
mat = require 'matio' --store in matlab format
cudnn.fastest = true
cudnn.benchmark = true 
require 'libmse'
require 'mse' --this takes care of including marginMSE and weightMSE
paths.dofile('fbcunn_files/AbstractParallel.lua')
paths.dofile('fbcunn_files/ModelParallel.lua')
paths.dofile('fbcunn_files/DataParallel.lua')
--paths.dofile('fbcunn_files/Optim.lua') --uncomment only when using nn.Optim
paths.dofile('funcs.lua')
paths.dofile('create_model.lua')
classSampling = torch.load('class_sampling.t7')

--------------------------------
-- Default Tensor Type is Float
--------------------------------
torch.setdefaulttensortype('torch.FloatTensor')

------------------------
-- Options for training
------------------------
local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)
print(opt)

-------------------------
-- Seed value and GPU ID
-------------------------
cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed) --setting up the seed value

-----------------------------
-- Saving all the parameters 
-----------------------------
print('==> Saving parameters/options to file: ', paths.concat(opt.save,'optionsCommandLine.txt'))
local command = 'mkdir -p '..opt.save 
os.execute(command)
--saveTable(opt,paths.concat(opt.save,'options_commandline.txt'))

-----------------------------------------------
-- All the Categories in alphabetical order
-----------------------------------------------
gClass_naming={'Affection','Anger','Annoyance','Anticipation','Aversion','Confidence','Disapproval','Disconnection','Disquietment','Doubt/Confusion','Embarrassment','Engagement','Esteem','Excitement','Fatigue','Fear','Happiness','Pain','Peace','Pleasure','Sadness','Sensitivity','Suffering','Surprise','Sympathy','Yearning'}
print("Category names defined: Affection, Anger, ..., Yearning")

-------------
-- Load data
-------------
TrainData = torch.load(opt.data .. 'DiscreteContinuousAnnotations26_train.t7')
TestData = torch.load(opt.data .. 'DiscreteContinuousAnnotations26_test.t7')
ValData = torch.load(opt.data .. 'DiscreteContinuousAnnotations26_val.t7')
BASE_IMAGE_FOLDER = opt.images 

-----------------------------------------
-- Generate smaller dataset (randomized)
-----------------------------------------
if mini_dataset == true then --choose/set this option at the beginning of the main.lua file
    print('It is a mini-dataset')
    train_samples = 312
    val_samples = 104
    test_samples = 104
    TrainData, ValData, TestData = generateMiniDataset(train_samples, val_samples, test_samples, TrainData, ValData, TestData)
end

---------------------------
-- Model Tensor parameters
---------------------------
NUM_CLASSES = 26
NUM_CLASSES_CONTINUOUS = 3
BATCH_SIZE = opt.batchSize
IMGSizes={};
IMGSizes.INPUT_LOAD_SIZE = 256
IMGSizes.INPUT_LARGEST_SIZE = 224
IMGSizes.BODY_SIZE = 224 
IMGSizes.NUM_CLASSES = NUM_CLASSES
IMGSizes.NUM_CLASSES_CONTINUOUS = NUM_CLASSES_CONTINUOUS
epoch = opt.epochNumber
iTestBatchSize = 2
nTest = #TestData
nVal = #ValData

--------------------------------------------------------
-- Generating Discrete labels (weighted and normalized)
--------------------------------------------------------
print("==> using " .. opt.discLabel .. ",1 labels")
opt.discScale = 1
ValLabels   = GenGTMultiLabels(ValData,NUM_CLASSES,opt.discScale) --multilabels for test and val 
TestLabels  = GenGTMultiLabels(TestData,NUM_CLASSES,opt.discScale) --opt.discScale = 1
TrainLabels = GenGTLabels(TrainData,NUM_CLASSES,opt.discLabel)

------------------------------------------------------------
-- Generating continuous labels (average of all annotators)
------------------------------------------------------------
print("==> Generate Continuous labels")
ValContinuous,Nconcepts = GenGTContinuousMulti(ValData) --mean values for test and val sets
         TestContinuous = GenGTContinuousMulti(TestData)
        TrainContinuous = GenGTContinuousMulti(TrainData)

--------------------------------------------------------
-- Normalizing continous Annotations to the range [0-1]
--------------------------------------------------------
ValContinuous   = torch.div(ValContinuous,opt.contNorm); --opt.contNorm = 10
TrainContinuous = torch.div(TrainContinuous,opt.contNorm);
TestContinuous  = torch.div(TestContinuous,opt.contNorm);

-------------------------------
-- Defining per-Classs Weights
-------------------------------
print('defining per-class weights:')
gClass_weights = getWeightNormalization(TrainLabels,NUM_CLASSES,opt.normWeights,opt.normFactor);
print(torch.view(gClass_weights,1,NUM_CLASSES));
print("==================================")
print("==> Train label Range: [" .. torch.min(ValContinuous:view(-1)) .. "," .. torch.max(ValContinuous:view(-1)) .. ']')
print("==================================") 