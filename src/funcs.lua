--[[
     Assortment of various functions used by the codebase
]]--

----------------------------------------------------------------------------
-- Find the best epoch and save the text file in the same folder
----------------------------------------------------------------------------
function findBestEpoch_joint(log_filename, folder_path, total_epochs)
    file_handle = io.open(folder_path .. '/' .. log_filename) --reads the files
    local count = 1 --to skip the title line
    
    value = torch.Tensor(total_epochs) --stores the rank-values
    
    if file_handle then
        for line in file_handle:lines() do
            local itr, cont_loss, auc, gauc, ap, val, ars, dom = unpack(line:split(" ")) --unpack turns a table like the one given (if you use the recommended version) into a bunch of separate variables
            if count ~= 1 then --to skip the title 
                local temp = (cont_loss*val*ars*dom)/(auc*gauc*ap) --stores the minimized values
                value[itr] = temp
            end
            count = count +1
        end
    end
    sorted_values, indexes = torch.sort(value) --sorting 
    
    result_filename = '0_best_epoch.log'
    f = io.open(folder_path .. '/' .. result_filename, 'w')
    f:write('lowest_Cost_rank = ',sorted_values[1])
    f:write('\n')
    f:write('corresponding_Epoch = ',indexes[1])
    f:close()
    print('lowest_Cost_rank:',sorted_values[1]) --printing
    print('corresponding_Epoch:',indexes[1])
end

--------------------------------------------------------------------
-- pass the log files to find the best epochs using the loss values
--------------------------------------------------------------------
function grabLossValues(filePath, epochs)
    file_handle = io.open(filePath) --reads the files
    local count = 1 --to skip the title line
    
    lossVector = torch.Tensor(epochs) --stores the loss-values
    
    if file_handle then
        for line in file_handle:lines() do
            local itr, loss = unpack(line:split(" ")) --unpacks the first two column values into the variables
            if count ~= 1 then --to skip the title 
                lossVector[count-1] = loss
            end
            count = count +1
        end
    end
    return lossVector
end

--------------------------------
-- Saves a table 't' into a file
--------------------------------
function saveTable(t, fileName)
	local output = io.open(fileName, "w")
	output:write("--------------------------------------------------\n")
    output:write("-------------Training Params----------------------\n")
    output:write("--------------------------------------------------\n")
	for k, v in pairs(t) do
        output:write(string.format("%s = %s\n", k, v))
    end
    output:write("--------------------------------------------------\n")
	output:close()
end

------------------------
--extracts a mini-table
------------------------
function subrange(t, first, last,red) 
    local sub = {}
    for i=first,last do
        sub[#sub + 1] = t[red[i]] --[dimention]
          --if i == 1 then print(t[red[i]]) end
    end
    return sub
end

---------------------------
-- generates a mini-dataset
---------------------------
function generateMiniDataset(train_samples, val_samples, test_samples, TrainData, ValData, TestData)
    --randomizing the indexes to select from
    miniTrain = torch.randperm(#TrainData)
    miniVal = torch.randperm(#ValData)
    miniTest = torch.randperm(#TestData)

    --capturing the mini-data
    TrainData = subrange(TrainData, 1, train_samples, miniTrain) --subrange function defined in supportFunctions.lua
    ValData = subrange(ValData, 1, val_samples, miniVal)
    TestData = subrange(TestData, 1, test_samples, miniTest)
    
    return TrainData, ValData, TestData
end


function torch.find(tensor, val, dim)
   local i = 1
   local indice = {}
   if dim then
      assert(tensor:dim() == 2, "torch.find dim arg only supports matrices for now")
      assert(dim == 2, "torch.find only supports dim=2 for now")
            
      local colSize, rowSize = tensor:size(1), tensor:size(2)
      local rowIndice = {}
      tensor:apply(function(x)     
            if x == val then
               table.insert(rowIndice, i)
            end
            if i == rowSize then
               i = 1
               table.insert(indice, rowIndice)
               rowIndice = {}
            else
               i = i + 1
            end
         end)
   else
      tensor:apply(function(x)
         if x == val then
            table.insert(indice, i)
         end
         i = i + 1
      end)
   end
   return indice
end

---------------------------------------------------------
-- calculates area under the curve and average-precision
---------------------------------------------------------
function CurveEvaluation(predictions, labels,classNames, nfalseneg, recallstep)
   -- we use the binary ones and the output
    require 'torch'
   local targets = labels:gt(0):byte();
   local iNumClasses=targets:size(2);
   local meters={};   
   local auc={};
   local tpr={};
   local fpr={};
   local ap={};
   for indC=1,iNumClasses do
     meters[indC]=tnt.AUCMeter();
     meters[indC]:reset();
     meters[indC]:add(predictions:narrow(2,indC,1):double(),targets:narrow(2,indC,1));
     local tauc, ttpr, tfpr = meters[indC]:value();
     --print(string.format("(AUC)-%s:\t%.3f",classNames[indC],tauc));
     auc[indC]=tauc;
     tpr[indC]=ttpr;
     fpr[indC]=tfpr;     
     local a=predictions:narrow(2,indC,1):transpose(2,1):double();
     local b=targets:narrow(2,indC,1):transpose(2,1):double();
     local trec, tprec, ftap, tsortind = precisionrecall(a,b*2-1, nfalseneg, recallstep);     
     --local trec, tprec, ftap, tsortind = precisionrecall(,targets:narrow(2,indC,1):double():transpose(2,1)*2-1);     
     ap[indC]=ftap[1];     
     --print(string.format("(PR)-%s:\t%.3f",classNames[indC],ftap[1]));
     print(string.format("%s:\t [AUC-%.3f ]\t [PR-%.3f ]\t ",classNames[indC],tauc,ftap[1]));
   end
   local gmeter=tnt.AUCMeter();
   gmeter:reset();
   gmeter:add(predictions:contiguous():view(-1):double(),targets:contiguous():view(-1));
   local gauc, gtpr, gfpr = gmeter:value();  
   --[[
   print("==> Global AUC ==")
   print(gauc);
   print("==> Averaged AUCs ==")
   local auc_avg=torch.Tensor(auc):mean();
   print(auc_avg);   
   print("==> Averaged AP (Precision-recall) ==")
   local ap_avg=torch.Tensor(ap):mean();
   print(ap_avg);
   print("==> Global AP (Precision-recall) ==")
   local gtrec, gtprec, gftap, gtsortind = precisionrecall(predictions:view(-1):double(),targets:view(-1):double()*2-1);     
   print(gftap);
   ]]--
   
   local auc_avg=torch.Tensor(auc):mean();
   local ap_avg=torch.Tensor(ap):mean();
   local gtrec, gtprec, gftap, gtsortind = precisionrecall(predictions:contiguous():view(-1):double(),targets:contiguous():view(-1):double()*2-1);     
   print(string.format("Global Area under curve:\t [AUC-%.3f ]\t [PR-%.3f ]",gauc,gftap));
   print(string.format("Average Area under curve:\t [AUC-%.3f ]\t [PR-%.3f ]",auc_avg,ap_avg));
   
   return auc_avg,gauc,ap_avg,gftap
end
  
  
-----------------------------------------------------------------
-- normalizing the weights in an attempt to un-bias the dataset 
-----------------------------------------------------------------
function getWeightNormalization(labelstocheck,iNUM_CLASSES,nW,nF)
    local histClasses=torch.sum(torch.gt(labelstocheck:float(),0):float(),1);--this is to make sure it works no matter what labels are we using.        
    --local normHist = histClasses / histClasses:sum();    --as we are then dividing, 
    local normHist = histClasses;--the lower number the larger the weight
    
    --local normHist = histClasses;    
    local class_weights = torch.Tensor(iNUM_CLASSES):fill(1);
    for i = 1, iNUM_CLASSES do
       if histClasses[1][i] < 1 then
          class_weights[i] = 0.0001;
       else
          --class_weights[i] = 1 / (torch.log(1.2 + normHist[1][i]));
         if nW > 0 then           
        --gClass_weights[i] = 1 / (torch.log(1.2 + normHist[1][i]));
            class_weights[i] = 1 / (torch.log(nF + normHist[1][i]));        
         else           
            class_weights[i] = 1 / (normHist[1][i]);
         end
       end
    end  
    return class_weights;
  end
  
-----------------------------------------------------------------
-- generates vectors that contain the continuous dimension labels
-----------------------------------------------------------------
function GenGTContinuous(GTData,iNconcepts)
  --to avoid the problem of random sorting in recursive calls (if any).
  local Nconcepts = iNconcepts or {'Valence','Arousal','Dominance'};
  local gtlabel=torch.Tensor(#GTData,#Nconcepts):fill(0);  
  for indAnn =1,#GTData do 
    --for key,val in pairs(ValData[1].workers[1].continuous) do print(key,val) end
    --local indV=1;
    --for key,val in pairs(Nconcepts) do 
    --  gtlabel[indAnn][indV]=GTData[indAnn].workers[1].continuous[key];
    --  indV=indV+1;
    --end
    for indV = 1,#Nconcepts do 
      gtlabel[indAnn][indV]=GTData[indAnn].workers[1].continuous[Nconcepts[indV]];    
    end
  end
  return gtlabel,Nconcepts;
end

----------------------------------------------------------------------
-- generates vectors that contain the continuous dimension labels
-- but also takes mean prediction of all the workers for Test and Validation sets
----------------------------------------------------------------------
function GenGTContinuousMulti(GTData,iNconcepts)
  --to avoid the problem of random sorting in recursive calls (if any).
  local Nconcepts = iNconcepts or {'Valence','Arousal','Dominance'};
  local gtlabel=torch.Tensor(#GTData,#Nconcepts):fill(0);  
  for indAnn =1,#GTData do 
    for indV = 1,#Nconcepts do 
      local temp = 0;
      for indWorker = 1, #GTData[indAnn].workers do
        temp = temp + GTData[indAnn].workers[indWorker].continuous[Nconcepts[indV]];    
      end
      gtlabel[indAnn][indV] = temp/(#GTData[indAnn].workers)
    end
  end
  --## Alternatively, one can use the following iterator method
    --[[
        for indAnn in pairs(GTData) do 
            gtlabel[indAnn][1] = GTData[indAnn].workers[1].continuous.Valence
            gtlabel[indAnn][2] = GTData[indAnn].workers[1].continuous.Arousal
            gtlabel[indAnn][3] = GTData[indAnn].workers[1].continuous.Dominance
        end
      ]]--  
  return gtlabel,Nconcepts;
end

---------------------------------------------------------------------------------------------------------------
--generates weighted categorical/discrete labels 
-- this generates weighted labels for test/val; given the no. of annotators it will give the weights accordingly
---------------------------------------------------------------------------------------------------------------
function GenGTMultiLabels(GTData,NClasses, scale) 
  local gtlabel=torch.FloatTensor(#GTData,NClasses):fill(0);  
    for indAnn =1,#GTData do 
      for indWorker = 1,#GTData[indAnn].workers do
        for indLabel=1,#GTData[indAnn].workers[indWorker].labels do
          gtlabel[indAnn][GTData[indAnn].workers[indWorker].labels[indLabel]] = gtlabel[indAnn][GTData[indAnn].workers[indWorker].labels[indLabel]] + scale*1
        end
      end
      gtlabel[indAnn]:div(gtlabel[indAnn]:max())  
    end
    return gtlabel;
  end

-------------------------------------------------
--generates weighted categorical/discrete labels
-------------------------------------------------
function GenGTLabels(GTData,NClasses,bgValue,scale)
  local ibgValue = bgValue or 0; --passed 0
  local iScale = scale or 1; --passed 10  
  
  --local gtlabel=torch.Tensor(#GTData,NClasses):fill(0);  
  local gtlabel=torch.Tensor(#GTData,NClasses):fill(ibgValue*iScale);
  for indAnn =1,#GTData do  
    --local filename = BASE_IMAGE_FOLDER .. ValData[indAnn].folder .. '/' ..  ValData[indAnn].filename;
    --local inputImage=image.load(filename);  
    for indLabel=1,#GTData[indAnn].workers[1].labels do    
      gtlabel[indAnn][GTData[indAnn].workers[1].labels[indLabel]]=1*iScale;
    end
  end
  return gtlabel;
end

----------------------------------------------------------------------
-- returns image and body crops according to their defined dimensions 
----------------------------------------------------------------------
function GetImagePatches(filename,GTAnnotation,verbose)
  local bVerbose=verbose or 0;
  local Sizes=verbose or {};     
  local input = image.load(filename) 
  local iW = input:size(3);
  local iH = input:size(2);
    
  if input:size(1) > 3 then  input = input[{{1,3},{},{}}]  end --reducing 4 dim to 3
  if input:dim() == 3 and input:size(1) == 1 then -- 1-channel image
      input = input:repeatTensor(3,1,1);
  end  

  --mean and std calculated from (datasets/emotic/simplified_meanStd_emotic.lua)  
  emotic_mean = {0.4709, 0.4409, 0.4062}
  emotic_std  = {0.2817, 0.2741, 0.2810}  

  for i=1,input:size(1) do -- channels
    if emotic_mean then input[{{i},{},{}}]:add(-emotic_mean[i]) end
    if emotic_std then input[{{i},{},{}}]:div(emotic_std[i]) end
  end

  if #GTAnnotation.body_bbox > 3 then
    body_x1=math.max(1,GTAnnotation.body_bbox[1])-1;
    body_y1=math.max(1,GTAnnotation.body_bbox[2])-1;
    body_x2=math.min(GTAnnotation.body_bbox[3],iW);
    body_y2=math.min(GTAnnotation.body_bbox[4],iH);
    inputBody = image.crop(input, body_x1,body_y1,body_x2,body_y2);
  else
    if bVerbose>0 then
      print("==> body annotation is missing, filling with zeros");
    end    
  end  
    
  inputBodySc  = image.scale(inputBody, Sizes.BODY_SIZE,Sizes.BODY_SIZE)
  inputFullL   = image.scale(input, Sizes.INPUT_LARGEST_SIZE,Sizes.INPUT_LARGEST_SIZE);
  
  return {inputFullL,inputBodySc};
end


------------------------------------------------------------
-- generates precision, recall and average-precision values 
------------------------------------------------------------
function precisionrecallmatrix(conf, labels, nfalseneg, recallstep)
   assert(conf:nDimension()==2)
   assert(labels:nDimension()==2)
   assert(conf:isSameSizeAs(labels))
   assert(nfalseneg==nil or nfalseneg:nDimension()==1)
   
   local nSamples = conf:size(2)
   local nClasses = conf:size(1)
   
   -- allocate
   local rec = torch.FloatTensor(nClasses, nSamples)
   local prec = torch.FloatTensor(nClasses, nSamples)
   local ap = torch.FloatTensor(nClasses)
   local sortind = torch.LongTensor(nClasses, nSamples)
   
   for i=1,nClasses do
      local _conf = conf:select(1,i)
      local _labels = labels:select(1,i)
      local _nfalseneg
      if nfalseneg then
         _nfalseneg = nfalseneg[i]
      end
      local _recallstep = recallstep
      local _rec, _prec, _ap, _sortind = precisionrecallvector(_conf, _labels, _nfalseneg, _recallstep)
      
      rec:select(1,i):copy(_rec)
      prec:select(1,i):copy(_prec)
      ap[i]=_ap
      sortind:select(1,i):copy(_sortind)
   end
   return rec, prec, ap, sortind
end

-- supporting function for the function 'precisionrecallmatrix' defined above
function precisionrecallvector(conf, labels, nfalseneg, recallstep)
   assert(conf:nDimension()==1)
   assert(labels:nDimension()==1)
   assert(conf:isSameSizeAs(labels))
   
   local nfalseneg = nfalseneg or 0
   local recallstep = recallstep or 0.1
   
   local so,sortind = conf:sort(true)
   
   local _tp=labels:index(1,sortind):gt(0):float()
   local _fp=labels:index(1,sortind):lt(0):float()
   local npos=labels:eq(1):float():sum() + nfalseneg
   
   -- precision / recall computation
   
   local tp = _tp:cumsum()
   local fp = _fp:cumsum()
   
   local rec = tp/npos
   local _fptp=fp+tp
   local prec = tp:cdiv(_fptp)
   
   -- ap calculation
   
   local ap = 0
   local recallpoints = 0
   local mask
   local p
      
   for i=0,1,recallstep do
      recallpoints=recallpoints+1
   end
   
   for i=0,1,recallstep do
      mask = rec:ge(i)
      if mask:max()>0 then
         p = prec:maskedSelect(mask):max()
      else
         p=0
      end
      ap = ap + p/recallpoints
   end   
   
   return rec, prec, ap, sortind
end

-- supporting function for the function 'precisionrecallmatrix' defined above
function precisionrecall(conf, labels, nfalseneg, recallstep)
   if conf:nDimension()==2 then
      return precisionrecallmatrix(conf, labels, nfalseneg, recallstep)
   elseif conf:nDimension()==1 then
      return precisionrecallvector(conf, labels, nfalseneg, recallstep)
   else
      error('vectors or matrices (classes x samples) expected')
   end
end

--------------------------
-- some utility functions
--------------------------
function getAnnotationByClass(class)
  local indAnnClass=1;
  --this is a table, accessing is simple:
  class2Sample = 'class_' .. class;
--  local index = math.ceil(torch.uniform() * self.classListSample[class]:nElement())  
  local index = math.ceil(torch.uniform() * classSampling[class2Sample]:nElement());
  --print(class2Sample)
  --print(index .. ' ' .. classSampling[class2Sample]:nElement());
  indAnnClass=classSampling[class2Sample][1][index];  
  return indAnnClass;
end

function getIndices(nSamples,iNumClasses)
 local shuffledind={};
 local numRounds=torch.ceil(nSamples/iNumClasses);
 if (numRounds*iNumClasses) == nSamples then -- is a multiple of the number of classes
   for indR=1,numRounds do
     for indC = 1,iNumClasses do
       table.insert(shuffledind,getAnnotationByClass(indC));
     end
   end
  return shuffledind,(numRounds*iNumClasses); 
else  
  --random class selection    
    for indR=1,nSamples do
         local indC=torch.ceil(torch.uniform(0,iNumClasses));
         table.insert(shuffledind,getAnnotationByClass(indC));       
     end
    return shuffledind,nSamples;     
  end
end