classdef convLayer
    %CONVLAYER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        % Four dimensions of W
        h; % height
        w; % width
        m; % input feature number
        n; % output feature number
        
        W; % Weight
        b; % bias
        activate; % activation type, 'relu' or 'none'
        pad;
        batchSize;
        input;
        output;
        
        lr; % learning rate
        wc; % weight decay
        mom; % momentum
        grad_W; % gradients of W
        grad_b; % gradients of b
        inc_W; % last update of W
        inc_b; % last update of b
        
        delta; % error
    end
    
    methods
        function layer = convLayer(h, w, m, n, normStd, activate, lr, wc, mom)  
            % Constructor function, initialize the object
            layer.h = h;
            layer.w = w;
            layer.m = m;
            layer.n = n;
            % pad the input to keep the size of output unchanged
            layer.pad = [floor((layer.h - 1) / 2), layer.h - 1 - floor((layer.h - 1) / 2), floor((layer.w - 1) / 2), layer.w - 1 - floor((layer.w - 1) / 2)];
            
            layer.W = single(random('norm', 0, normStd, h, w, m, n));
            layer.b = zeros(n, 1, 'single');
            layer.inc_W = zeros(size(layer.W), 'single');
            layer.inc_b = zeros(size(layer.b), 'single');
            layer.activate = activate;
            layer.lr = lr;
            layer.wc = wc;
            layer.mom = mom;
        end
        
        function layer = forward(layer, input)
            assert(size(input, 3) == layer.m);
            layer.batchSize = size(input, 4);
            
            layer.input = input;
            % convolution
            layer.output = vl_nnconv(layer.input, layer.W, layer.b, 'pad', layer.pad);
            % activation
            if strcmp(layer.activate, 'relu')
                layer.output = layer.output .* (layer.output > 0);
            end
            if strcmp(layer.activate, 'sigmoid')
                layer.output =1./(1+exp(-layer.output));
            end
        end
        
        function layer = backward(layer, delta)
            assert(isequal(size(delta), size(layer.output)));
            
            % compute activation gradient
            if strcmp(layer.activate, 'relu')
                delta = delta .* (layer.output > 0);
            end
            if strcmp(layer.activate, 'sigmoid')
                delta = delta .*layer.output.*(1-layer.output);
            end
            
            % compute convolution gradient
            [layer.delta, layer.grad_W, layer.grad_b] = vl_nnconv(layer.input, layer.W, layer.b, delta, 'pad', layer.pad);
            
            % update
            layer.inc_W = layer.mom * layer.inc_W - layer.lr * (layer.grad_W + layer.wc * layer.W);
            layer.W = layer.W + layer.inc_W;
            layer.inc_b = layer.mom * layer.inc_b - 2 * layer.lr * layer.grad_b;
            layer.b = layer.b + layer.inc_b;
        end
    end
end

