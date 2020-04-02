
function varargout = guidemo(varargin)
% GUIDEMO MATLAB code for guidemo.fig
%      GUIDEMO, by itself, creates a new GUIDEMO or raises the existing
%      singleton*.
%
%      H = GUIDEMO returns the handle to a new GUIDEMO or the handle to
%      the existing singleton*.
%
%      GUIDEMO('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GUIDEMO.M with the given input arguments.
%
%      GUIDEMO('Property','Value',...) creates a new GUIDEMO or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before guidemo_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to guidemo_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help guidemo

% Last Modified by GUIDE v2.5 21-Mar-2020 17:25:50

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @guidemo_OpeningFcn, ...
                   'gui_OutputFcn',  @guidemo_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before guidemo is made visible.
function guidemo_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to guidemo (see VARARGIN)

% Choose default command line output for guidemo
handles.output = hObject;
label='0'
       set(handles.edit1,'String',label);
       
a=ones([256 256]);
axes(handles.axes1);
imshow(a);

axes(handles.axes2);
imshow(a);

axes(handles.axes3);
imshow(a);

axes(handles.axes4);
imshow(a);

axes(handles.axes5);
imshow(a);
% Update handles structure
guidata(hObject, handles);

% UIWAIT makes guidemo wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = guidemo_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in Browse.
function Browse_Callback(hObject, eventdata, handles)
% hObject    handle to Browse (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

 [filename, pathname] = uigetfile('*.*', 'Pick a MATLAB code file');
    if isequal(filename,0) || isequal(pathname,0)
       disp('User pressed cancel')
    else
       filename=strcat(pathname,filename);
       axes(handles.axes1);
       im=imread(filename);
       im = readAndPreprocessImage(filename)
       imshow(im);
       handles.im=im;
       label='0'
       set(handles.edit3,'String',label);
       set(handles.edit1,'String',label);
    end

% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in pushbutton8.
function pushbutton8_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
cd Database2
delete *.mat;
Db_features = [];
% L = waitbar(0,'Please Wait...Processing');
% 
for it = 1:128
Inp = strcat(int2str(it),'.jpg');
Dinp = imread(Inp);
Dinp = imresize(Dinp,[256 256]);
    [r c p]=size(Dinp);
    if p == 3
     Dinp = rgb2gray(Dinp);
    end

%%%%%%%% Feature Extraction using multiwavelet transform
%%%%%%%% and cooccurrence features

% % % % % 1 level decomp

    [A1 H1 V1 D1] = dwt2(Dinp,'db1');

% % % % 2nd level decomp
    [A2 H2 V2 D2] = dwt2(A1,'db1');

% % % 3rd level Decomp

    [A3 H3 V3 D3] = dwt2(A2,'db1');

% % % 4th level Decomp

    [A4 H4 V4 D4] = dwt2(A3,'db1'); 
    
    % % % Select the wavelet coefficients oly H4 and V4
    % % % GLCM features for H4

    H4 = uint8(H4);
    Min_val = min(min(H4));
    Max_val = max(max(H4));
    level = round(Max_val - Min_val);
    GLCM = graycomatrix(H4,'GrayLimits',[Min_val Max_val],'NumLevels',level);
    stat_feature = graycoprops(GLCM);
    Energy_fet1 = stat_feature.Energy;
    Contr_fet1 = stat_feature.Contrast;
    Corrla_fet1 = stat_feature.Correlation;
    Homogen_fet1 = stat_feature.Homogeneity;
    % % % % % Entropy
            R = sum(sum(GLCM));
            Norm_GLCM_region = GLCM/R;

            Ent_int = 0;
            for k = 1:length(GLCM)^2
                if Norm_GLCM_region(k)~=0
                    Ent_int = Ent_int + Norm_GLCM_region(k)*log2(Norm_GLCM_region(k));
                end
            end
    % % % % % % Ent_int = entropy(GLCM);
            Entropy_fet1 = -Ent_int;

    V4 = uint8(V4);
    Min_val = min(min(V4));
    Max_val = max(max(V4));
    level = round(Max_val - Min_val);
    GLCM = graycomatrix(V4,'GrayLimits',[Min_val Max_val],'NumLevels',level);
    stat_feature = graycoprops(GLCM);
    Energy_fet2 = stat_feature.Energy;
    Contr_fet2 = stat_feature.Contrast;
    Corrla_fet2= stat_feature.Correlation;
    Homogen_fet2 = stat_feature.Homogeneity;
    % % % % % Entropy
            R = sum(sum(GLCM));
            Norm_GLCM_region = GLCM/R;

            Ent_int = 0;
            for k = 1:length(GLCM)^2
                if Norm_GLCM_region(k)~=0
                    Ent_int = Ent_int + Norm_GLCM_region(k)*log2(Norm_GLCM_region(k));
                end
            end
    % % % % % % Ent_int = entropy(GLCM);
            Entropy_fet2 = -Ent_int;
            F1 = [Energy_fet1 Contr_fet1 Corrla_fet1 Homogen_fet1 Entropy_fet1];
            F2 = [Energy_fet2 Contr_fet2 Corrla_fet2 Homogen_fet2 Entropy_fet2];
            Fet = [F1 F2]';
            
     Db_features = [Db_features Fet];                 

%      waitbar(it/15,L);

end 
% close(L);
disp('Database Images Features :');
disp(Db_features);
cd ..
handles.Dfeatures = Db_features;
guidata(hObject,handles);
helpdlg('Database successfully added');

% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
im = handles.im;
%input = imresize(Freg,0.5, 'nearest');
%input = imresize(Freg,[480 640]);
%pcrop = imcrop(handles.img,[180 180 250 250]);

inp=imresize(im,[256 256]);
 
   if size(inp,3)>1
     inp = rgb2gray(inp);
   end
%    cd ..
   axes(handles.axes2);
   imshow(inp);
   title('Test Image');
inp=medfilt2(inp);
axes(handles.axes3);
imshow(inp);
title('Filtered image')
handles.inp = inp;

% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
inp=handles.inp;
[rows columns numberOfColorBands] = size(inp)
[nonZeroRows nonZeroColumns] = find(inp);

topRow = min(nonZeroRows(:));
bottomRow = max(nonZeroRows(:));
leftColumn = min(nonZeroColumns(:));
rightColumn = max(nonZeroColumns(:));
% Extract a cropped image from the original.
croppedImage = inp(topRow:bottomRow, leftColumn:rightColumn);
axes(handles.axes4);
imshow(croppedImage);
handles.croppedImage = croppedImage;
disp('ROI Calculated')
% Update handles structure
guidata(hObject, handles);

% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
inp=handles.inp;
out=Lclustering(inp)
handles.out = out;
disp('Segmentation completed');
% Update handles structure
guidata(hObject, handles);


% --- Executes on button press in Classify.
function Classify_Callback(hObject, eventdata, handles)
% hObject    handle to Classify (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

load('trainedNet1.mat');

im=handles.im;
sIz=size(im)
axes(handles.axes2);
imshow(im);

PreprocessesImages=imresize(im,[227 227]);
PreprocessesImages=uint8(PreprocessesImages);
axes(handles.axes2);
imshow(PreprocessesImages);
im=PreprocessesImages;



label = char(classify(net,im)); % classify with deep learning
set(handles.edit1,'String',label);

if strcmp(label,'Normal')
    
    label='Abnormal Detected';
    set(handles.edit3,'String',label);

else
    label='Normal Detected';
    set(handles.edit3,'String',label);
    
end


% --- Executes on button press in Trainning.
function Trainning_Callback(hObject, eventdata, handles)
% hObject    handle to Trainning (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
loc=pwd;
location = strcat(loc,'\nordatabase');
imds = imageDatastore(location,'IncludeSubfolders',1,...
    'LabelSource','foldernames');
tbl = countEachLabel(imds);
a='cnn.mat'
net=getcnn(a);
net.Layers

%% Perform net 
% Get the layers from the network. the layers define the network
% architecture and contain the learned weights.
layers = net.Layers(1:end-3);

% %optional, Add new fully connected layer for 2 categories.
layers(end+1) = fullyConnectedLayer(64, 'Name', 'special_2');
layers(end+1) = reluLayer;


layers(end+1) = fullyConnectedLayer(height(tbl), 'Name', 'fc8_2')

layers(end+1) = softmaxLayer
layers(end+1) = classificationLayer()

% Modify image layer to add randcrop data augmentation. This increases the
% diversity of training images. The size of the input images is set to the
% original networks input size.
layers(1) = imageInputLayer([227 227 3]);


%% Setup learning rates for fine-tuning
% fc 8 - bump up learning rate for last layers
layers(end-2).WeightLearnRateFactor = 10;
layers(end-2).WeightL2Factor = 1;
layers(end-2).BiasLearnRateFactor = 20;
layers(end-2).BiasL2Factor = 0;

%% Equalize number of images of each class in training set
minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category
%Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount);

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)
[trainingDS, testDS] = splitEachLabel(imds, 0.8,'randomize');
%  trainingDS=imds;
% Convert labels to categoricals
trainingDS.Labels = categorical(trainingDS.Labels);
trainingDS.ReadFcn = @readFunctionTrain;

% Setup test data for validation
testDS.Labels = categorical(testDS.Labels);
testDS.ReadFcn = @readFunctionValidation;

%% Fine-tune the Network

miniBatchSize = 128; % lower this if your GPU runs out of memory.
numImages = numel(trainingDS.Files);

% Run training for 5000 iterations. Convert 20000 iterations into the
% number of epochs this will be.
maxEpochs=get(handles.edit4,'String')
maxEpochs=str2num(maxEpochs);
%maxEpochs =300; % one complete pass through the training data
% batch size is the number of images it processes at once. Training
% algorithm chunks into manageable sizes. 

lr=get(handles.edit2,'String');
lr=str2num(lr);
% lr = 0.0001;
opts = trainingOptions('sgdm', ...
    'LearnRateSchedule', 'none',...
    'InitialLearnRate', lr,... 
    'MaxEpochs', maxEpochs, ...
     'MiniBatchSize', miniBatchSize,...
    'Plots','training-progress');
   
analyzeNetwork(layers)

net = trainNetwork(trainingDS, layers, opts);
save('trainedNet1.mat','net')
% This could take over an hour to run, so lets stop and load a pre-traiend
% version that used the same data
warndlg('Training Completed');


function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit4_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit4 as text
%        str2double(get(hObject,'String')) returns contents of edit4 as a double


% --- Executes during object creation, after setting all properties.
function edit4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

inp = handles.inp;

[LL LH HL HH] = dwt2(inp,'db3');

aa = [LL LH;HL HH];



axes(handles.axes5);
imshow(aa,[]);
title('Discrete wavelet transform');

% % % Select the wavelet coefficients LH3 and HL3
% % % Haralick features for LH3

LH3 = uint8(LL);
Min_val = min(min(LH3));
Max_val = max(max(LH3));
level = round(Max_val - Min_val);
GLCM = graycomatrix(LH3,'GrayLimits',[Min_val Max_val],'NumLevels',level);
stat_feature = graycoprops(GLCM);
Energy_fet1 = stat_feature.Energy;
Contr_fet1 = stat_feature.Contrast;
Corrla_fet1 = stat_feature.Correlation;
Homogen_fet1 = stat_feature.Homogeneity;

% % % % % Entropy
        R = sum(sum(GLCM));
        Norm_GLCM_region = GLCM/R;
        
        Ent_int = 0;
        for k = 1:length(GLCM)^2
            if Norm_GLCM_region(k)~=0
                Ent_int = Ent_int + Norm_GLCM_region(k)*log2(Norm_GLCM_region(k));
            end
        end
        Entropy_fet1 = -Ent_int;



%%%%% Feature Sets

F1 = [Energy_fet1 Contr_fet1 Corrla_fet1 Homogen_fet1 Entropy_fet1];

qfeat = [F1 ]';
save qfeat qfeat;

disp('Query Features: ');
disp(qfeat);
helpdlg('GLCM - Texture Features Extracted');


% Update handles structure
guidata(hObject, handles);
