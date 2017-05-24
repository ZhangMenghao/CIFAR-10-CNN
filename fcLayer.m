classdef fcLayer
    %CONVLAYER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        n; % output feature number
        m; % input feature number
        
        W; % Weight
        b; % bias
        activate;
        batchSize;
        input;
        output;
        
        % refer to the comments in convLayer
        lr;
        wc;
        mom;
        grad_W;
        grad_b;
        inc_W;
        inc_b;
        delta;
    end
    
    methods
        function layer = fcLayer(n, m, normStd, activate, lr, wc, mom)
            % Constructor function
            layer.n = n;
            layer.m = m;
            layer.W = single(random('norm', 0, normStd, n, m));
            layer.b = zeros(n, 1, 'single');
            layer.inc_W = zeros(size(layer.W), 'single');
            layer.inc_b = zeros(size(layer.b), 'single');
            layer.activate = activate;
            layer.lr = lr;
            layer.wc = wc;
            layer.mom = mom;
        end
        
        function layer = forward(layer, input)
            % Insert your code
            
            layer.input=input;
			inputs = reshape(layer.input,layer.m,100);
            layer.output=layer.W*inputs;
            bs=repmat(layer.b,1,100);
            layer.output=layer.output+bs;
            
            if strcmp(layer.activate, 'relu')
                layer.output = layer.output .* (layer.output > 0);
            end
              
        end
        
        function layer = backward(layer, delta)
            
            if strcmp(layer.activate, 'relu')
                delta = delta .* (layer.output > 0);
            end
            layer.grad_W=zeros(size(layer.W), 'single');
            layer.grad_b=zeros(size(layer.b), 'single');
            inputs = reshape(layer.input,layer.m,100);
            for i=1:1:100
                layer.grad_W=layer.grad_W+delta(:,i)*inputs(:,i)';
                layer.b=layer.b+delta(:,i);
            end
            layer.grad_W=layer.grad_W./100;
            layer.b=layer.b./100;
           
            layer.delta=(layer.W)'*delta;
            
            layer.delta = reshape(layer.delta,size(layer.input));
            
            layer.inc_W = layer.mom * layer.inc_W - layer.lr * (layer.grad_W + layer.wc * layer.W);
            layer.W = layer.W + layer.inc_W;
            layer.inc_b = layer.mom * layer.inc_b - 2 * layer.lr * layer.grad_b;
            layer.b = layer.b + layer.inc_b;
            % Insert your code
        end
    end
end