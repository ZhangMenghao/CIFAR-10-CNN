% Load training set and validation set
clear all;
dataDir = 'cifar-10-batches-mat';
load(fullfile(dataDir, 'data_batch_1.mat'));
data_train = data;
label_train = labels;
for i = 2:5
    load(fullfile(dataDir, ['data_batch_' num2str(i) '.mat']));
    data_train = [data_train; data];
    label_train = [label_train, labels];
end
label_train = label_train + 1; % original label ranges from 0 to 9, we want 1 to 10.

load(fullfile(dataDir, 'data_batch_5.mat'));
data_valid = data;
label_valid = labels;
label_valid = label_valid + 1; % 0 to 9 ---> 1 to 10

% Define the CNN
% In this example, weight decay, momentum and dropout are NOT used.
conv1 = convLayer(5, 5, 3, 32, 0.1, 'relu', 0.001, 0.000, 0.0); %?'sigmoid'??'relu'???????????
pool1 = poolLayer('max', 3, 2, [0, 1, 0, 1]); 
conv2 = convLayer(5, 5, 32, 32, 0.01, 'relu', 0.001, 0.000, 0.0); 
pool2 = poolLayer('avg', 3, 2, [0, 1, 0, 1]);
conv3 = convLayer(5, 5, 32, 64, 0.01, 'relu', 0.001, 0.000, 0.0);
pool3 = poolLayer('avg', 3, 2, [0, 1, 0, 1]);
fc1 = fcLayer(64, 1024, 0.01, 'none', 0.001, 0.00, 0.0);
fc2 = fcLayer(10, 64, 0.01, 'none', 0.001, 0.00, 0.0); 
softmax = softmaxLayer;

% Some parameters used in training
N_img = 50000;
N_epoch = 10 ;
batchSize = 100;
N_batch = N_img / batchSize;
% Preprocess the data by substracting the mean value
data_mean = permute(reshape(repmat(mean(single(data_train), 1), [batchSize, 1]), [batchSize, 32, 32, 3]), [3, 2, 4, 1]);

for epoch = 1:N_epoch  
    if epoch == 9
        conv1.lr = conv1.lr * 0.1;
        conv2.lr = conv2.lr * 0.1;
        conv3.lr = conv3.lr * 0.1;
        fc1.lr = fc1.lr * 0.1;
        fc2.lr = fc2.lr * 0.1;
    end
    if epoch == 10
        conv1.lr = conv1.lr * 0.1;
        conv2.lr = conv2.lr * 0.1;
        conv3.lr = conv3.lr * 0.1;
        fc1.lr = fc1.lr * 0.1;
        fc2.lr = fc2.lr * 0.1;
    end
    
    tic;
    loss = 0;
    accuracy = 0;
    for batch = 1:N_batch
        img = single(permute(reshape(data_train(batchSize * (batch - 1) + 1:batchSize * batch, :), [batchSize, 32, 32, 3]), [3, 2, 4, 1]));
        img = img - data_mean;
        label = label_train(batchSize * (batch - 1) + 1:batchSize * batch);
        
        % Feedforward
        conv1 = forward(conv1, img);
        pool1 = forward(pool1, conv1.output);
        conv2 = forward(conv2, pool1.output);
        pool2 = forward(pool2, conv2.output);
        conv3 = forward(conv3, pool2.output);
        pool3 = forward(pool3, conv3.output);
        fc1 = forward(fc1, pool3.output);
        fc2 = forward(fc2, fc1.output);
        
        softmax = forward(softmax, fc2.output);
        
        % Backward
        softmax = backward(softmax, label);
        fc2 = backward(fc2, softmax.delta);
        fc1 = backward(fc1, fc2.delta);
        pool3 = backward(pool3, fc1.delta);
        conv3 = backward(conv3, pool3.delta);
        pool2 = backward(pool2, conv3.delta);
        conv2 = backward(conv2, pool2.delta);
        pool1 = backward(pool1, conv2.delta);
        conv1 = backward(conv1, pool1.delta);
        
        loss = loss + softmax.loss;
        accuracy = accuracy + softmax.accuracy;
        
        if mod(batch, 5) == 0
             fprintf(['Epoch ' num2str(epoch) ' batch ' num2str(batch) ' loss is ' num2str(loss / 5) ' accuracy is ' num2str(accuracy / 5) ' elapsed time is ' num2str(toc) ' seconds.\n']);
            loss = 0;
            accuracy = 0;
            
            % Validation
            tic;
            img = single(permute(reshape(data_valid(batchSize * (batch / 5 - 1) + 1:batchSize * batch / 5, :), [batchSize, 32, 32, 3]), [3, 2, 4, 1]));
            img = img - data_mean;
            label = label_valid(batchSize * (batch / 5 - 1) + 1:batchSize * batch / 5);
            
            conv1 = forward(conv1, img);
            pool1 = forward(pool1, conv1.output);
            conv2 = forward(conv2, pool1.output);
            pool2 = forward(pool2, conv2.output);
            conv3 = forward(conv3, pool2.output);
            pool3 = forward(pool3, conv3.output);
            fc1 = forward(fc1, pool3.output);
            fc2 = forward(fc2, fc1.output);
            softmax = forward(softmax, fc2.output);
            
            softmax = backward(softmax, label);
            fprintf(['Epoch ' num2str(epoch) ' Valid batch ' num2str(batch / 5) ' loss is ' num2str(softmax.loss) ' accuracy is ' num2str(softmax.accuracy) ' elapsed time is ' num2str(toc) ' seconds.\n']);
            tic;
        end
    end
end

% Load test data
load(fullfile(dataDir, 'test_batch.mat'));
data_test = data;
label_test = labels;
label_test = label_test + 1;
N_batch = 10000 / batchSize;
accuracy = 0;
% Testing
for batch = 1:N_batch
    img = single(permute(reshape(data_test(batchSize * (batch - 1) + 1:batchSize * batch, :), [batchSize, 32, 32, 3]), [3, 2, 4, 1]));
    img = img - data_mean;
    label = label_test(batchSize * (batch - 1) + 1:batchSize * batch);
    
    conv1 = forward(conv1, img);
    pool1 = forward(pool1, conv1.output);
    conv2 = forward(conv2, pool1.output);
    pool2 = forward(pool2, conv2.output);
    conv3 = forward(conv3, pool2.output);
    pool3 = forward(pool3, conv3.output);
    fc1 = forward(fc1, pool3.output);
    fc2 = forward(fc2, fc1.output);
    softmax = forward(softmax, fc2.output);
    
    softmax = backward(softmax, label);
    accuracy = accuracy + softmax.accuracy;
end
accuracy = accuracy / N_batch;
fprintf(['Test accuracy is ' num2str(accuracy) ', Training is done.\n']);