--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
--- This code does inference for single images, based on PAMI version of the model 
--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 

--load the required libraries 
require 'nn'
require 'cunn'
require 'cudnn'
require 'image'
require 'paths'
mat = require 'matio'

--define the paths for code, data and results
path = {}
path.base = '/root/capsule'
path.code = paths.concat(path.base, 'code')
path.data = paths.concat(path.base, 'data')
path.results = paths.concat(path.base, 'results')

--set the default tensor type
torch.setdefaulttensortype('torch.DoubleTensor')

--adding some basic functions
dofile(path.code .. '/funcs.lua')  -- https://github.com/rkosti/emotic/blob/master/funcs.lua

--load the model 
net = torch.load(path.data .. '/emoticCNN_model.t7') --baseline CNN for pami version (https://1drv.ms/u/s!AkYHbdGNmIVCgbYSIcSFYJgcApIRKw?e=slJTZp)

--switching to testing/evaluation mode
net:evaluate() 

--variable definitions
IMGSizes={};
IMGSizes.INPUT_LARGEST_SIZE = 224
IMGSizes.BODY_SIZE = 128 
emotic_classes = {'Affection','Anger','Annoyance','Anticipation','Aversion','Confidence','Disapproval','Disconnection','Disquietment','Doubt/Confusion','Embarrassment','Engagement','Esteem','Excitement','Fatigue','Fear','Happiness','Pain','Peace','Pleasure','Sadness','Sensitivity','Suffering','Surprise','Sympathy','Yearning'}

--load the thresholds calculated on validation set 
th = mat.load(path.data .. '/thresholds.mat', 'thresholds') --https://github.com/rkosti/emotic/blob/master/thresholds.mat

-- Load the image and the bounding-box data to be predicted
filename = path.data .. '/woman.jpg'
data_image = {}
data_image.body_bbox  = {11, 12, 738, 836} --format {x1, y1, x2, y2} 
input = image.load(filename)

ImgPatches = GetImagePatches(filename,data_image,IMGSizes)
print(itorch.image(ImgPatches[1]))
print(itorch.image(ImgPatches[2]))

--Image
x_img = torch.DoubleTensor(2, 3, IMGSizes.INPUT_LARGEST_SIZE, IMGSizes.INPUT_LARGEST_SIZE)
x_img = x_img:cuda()

--BODY
x_body = torch.DoubleTensor(2, 3, IMGSizes.BODY_SIZE, IMGSizes.BODY_SIZE)
x_body = x_body:cuda(); --input body crop of the person in the image - empty Tensor

x_img[1]  = ImgPatches[1]:cuda() --because only mini-batches supported
x_img[2]  = ImgPatches[1]:cuda() 
x_body[1] = ImgPatches[2]:cuda() 
x_body[2] = ImgPatches[2]:cuda()

-- Features 
inputData =  {x_img,x_body}
preds = net:forward(inputData)   

-- Discrete Predictions 
pred_classes_ind = torch.gt(preds[2][1]:double(), th)

-- Continuous predictions 
print('Predictions')
print('-----------------------------------')
print('Continuous Dimension levels')
print('-----------------------------------')
print(string.format('Valence: %.1f', preds[1][1][1]))
print(string.format('Arousal: %.1f', preds[1][1][2]))
print(string.format('Dominance: %.1f', preds[1][1][3]))
print('-----------------------------------')
-- Discrete Predictions 
pred_classes_ind = torch.gt(preds[2][1]:double(), th)

print('-----------------------------------')
print('Emotion Categories')
print('-----------------------------------')

for i=1,#emotic_classes do 
    if pred_classes_ind[i] == 1 then 
        print(string.format('%s ',  emotic_classes[i])) 
    end 
end

