clear all; clc; delete all

%% LOAD DATA
% file of the dataset
% datasetfolder='DATA\CLASSIFIER\SITTING_ALLFEAT\Table_Classifier';
% load 
load(fullfile(cd,'Table_Classifier3_YSIT_P.mat'));

% input variables observation x input[N I]
dataI=table2array(DataSET(:,1:end-1));        
% output labels observations x 1 [N 1]
dataO=table2array(DataSET(:,end));                          

%% DEFINE SUBJECTS
% subjects in dataset
SBJ=unique(DIVISOR);        % DIVISOR contain labels of subjects in DataSET

%% CHOOSE NUMBER OF FOLDS
% loop flag
loop=0;
while loop==0
flag=input('What validation model you would like to implement?\n   1 - 5 folds\n   2 - 10 folds\n   3 - LOSOCV\n\n');
    switch flag
        % 5 fold CV
        case 1
            folds = 6;
            loop = 1;
        
        % 10 fold CV
        case 2
            folds = 11;
            loop = 1;
        
        % LOSOCV
        case 3
            folds = length(SBJ);
            loop = 1;
            
        otherwise
            loop = 0;
    end
end

%% CHOOSE CLASSIFIER
loop=0;
while loop==0
flag=input('What ML model you would like to implement?\n   1 - KNN\n   2 - SVMpoly\n   3 - SVMgauss\n\n','s');                                           
    switch flag
        % SVM
        case 'SVMpoly'
            % Parameters of SVM to tune:
            % Kernel function
            % kernel=optimizableVariable('kernel',{'polynomial','gaussian'},'Type','categorical');
            % Box constraint
            box = optimizableVariable('box',[1e-5,1e5],'Transform','log'); 
            % Polynomial order
            order = optimizableVariable('order',[1,3],'Type','integer');
            % Kernel's scale
            % scale = optimizableVariable('scale',[3,64],'Type','integer');
            loop = 1;
        % KNN
        case 'SVMgauss'
            box = optimizableVariable('box',[1e-5,1e5],'Transform','log');
            scale = optimizableVariable('scale',[3,64],'Type','integer');
            loop = 1;
        case 'KNN'
            % Parameters of SVM to tune
            % Number of point
            neighbors=optimizableVariable('neighbors',[1,100],'Type','integer');
            % Distance
            distance=optimizableVariable('distance',{'euclidean','cityblock','chebychev','hamming'},'Type','categorical');
            % Weights
            weight=optimizableVariable('weight',{'Equal','Inverse'},'Type','categorical');
            loop = 1;

        otherwise
            loop = 0;
    end
end                                       

%% DEFINE SUBJECTS FOLDS
Shuffled=SBJ(randperm(numel(SBJ)));
step=floor(numel(SBJ)/folds);
division=1:step:(step*folds);
R=rem(numel(SBJ),folds);
subgroup=cell(folds,1);
models=cell(folds,1);
for k=1:folds
    subgroup{k}=Shuffled(division(k):division(k)+step-1);
end
if R~=0
    Remain=Shuffled(end-(R-1):end);
    subgroup=Redistribute(DIVISOR,subgroup,Remain);
end
% TEST FOLD CICLE
for t=1:folds
    indexes=cell(folds-1,1);
    i=1;
    logicalindex=cellfun(@(x)contains(DIVISOR,x),subgroup{t},'un',0);
    INDT=any(horzcat(logicalindex{:}),2);
    test=DataSET(INDT==1,:);
    dvlp=DataSET(INDT==0,:);
    DIVISORVal=DIVISOR(INDT==0);
    DIVISORTst=DIVISOR(INDT==1);
    disp('Develop Set Subjects\n');
    disp(unique(DIVISORVal));
    disp('-------------------');
    disp('Test Set Subjects\n');
    disp(unique(DIVISOR(INDT==1)))
    % VALIDATION FOLD DEFINITION
    for v=1:folds
        if v~=t
            logicalindex=cellfun(@(x)contains(DIVISORVal,x),subgroup{v},'un',0);
            IND=any(horzcat(logicalindex{:}),2);
            indexes{i}=find(IND==1);
            i=i+1;
        end
    end
    fun = @(x)CrossValidatedClassifier(dvlp,indexes,flag,x);
    switch flag
        % SVM
        case 'SVMpoly'
            % results = bayesopt(fun,[kernel,box,order,scale],'ConditionalVariableFcn',@condvariablefcn);
            results = bayesopt(fun,[box,order]);
        case 'SVMgauss'
            % results = bayesopt(fun,[kernel,box,order,scale],'ConditionalVariableFcn',@condvariablefcn);
            results = bayesopt(fun,[box,scale]);
        case 'KNN'
            results = bayesopt(fun,[neighbors,distance,weight]);
    end
    model.confusion.target=zeros(5,size(test.Response,1));
    model.confusion.output=zeros(5,size(test.Response,1));
    switch flag
        case 'SVMpoly'
            y.box=results.XAtMinObjective.box;
            y.order=results.XAtMinObjective.order;
            model.class=ImplementClassifier(dvlp(:,1:end),flag,y);
        case 'SVMgauss'
            y.box=results.XAtMinObjective.box;
            y.scale=results.XAtMinObjective.scale;
            model.class=ImplementClassifier(dvlp(:,1:end),flag,y);
        case 'KNN'
            y.neighbors=results.XAtMinObjective.neighbors;
            y.weight=results.XAtMinObjective.weight;
            y.distance=results.XAtMinObjective.distance;
            model.class=ImplementClassifier(dvlp(:,1:end),flag,y);
    end
    
    
    
    
    
    [predictions,scores]=predict(model.class,test(:,1:end-1));
    model.confusion.output=zeros(5,size(test.Response,1));
    model.confusion.output(1,predictions==1)=1;
    model.confusion.output(2,predictions==2)=1;
    model.confusion.output(3,predictions==3)=1;
    model.confusion.output(4,predictions==4)=1;
    model.confusion.output(5,predictions==5)=1;
    model.confusion.target=zeros(5,size(test.Response,1));
    model.confusion.target(1,test.Response==1)=1;
    model.confusion.target(2,test.Response==2)=1;
    model.confusion.target(3,test.Response==3)=1;
    model.confusion.target(4,test.Response==4)=1;
    model.confusion.target(5,test.Response==5)=1;
    [model.ROC.REST.x,model.ROC.REST.y,~,model.ROC.REST.auc]=perfcurve(test.Response,scores(:,1),1);
    [model.ROC.TL.x,model.ROC.TL.y,~,model.ROC.TL.auc]=perfcurve(test.Response,scores(:,2),2);
    [model.ROC.STAND.x,model.ROC.STAND.y,~,model.ROC.STAND.auc]=perfcurve(test.Response,scores(:,3),3);
    [model.ROC.BAL.x,model.ROC.BAL.y,~,model.ROC.BAL.auc]=perfcurve(test.Response,scores(:,4),4);
    [model.ROC.SIT.x,model.ROC.SIT.y,~,model.ROC.SIT.auc]=perfcurve(test.Response,scores(:,5),5);
    model.optimization=results;  
    model.TstSET=unique(DIVISORTst);
    model.ValSET=unique(DIVISORVal);
    models{t}=model;
   
end
               
    
        
        
        
        