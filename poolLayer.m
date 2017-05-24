classdef poolLayer
    %POOLLAYER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        method;
        poolSize;
        stride;
        pad;
        input;
        output;
        delta;
    end
    
    methods
        function layer = poolLayer(method, poolSize, stride, pad)
            layer.method = method;
            layer.poolSize = poolSize;
            layer.stride = stride;
            layer.pad = pad;
        end
        
        function layer = forward(layer, input)
            layer.input = input;
            layer.output = vl_nnpool(layer.input, layer.poolSize, 'stride', layer.stride, 'pad', layer.pad, 'method', layer.method);
        end
        
        function layer = backward(layer, delta)
            assert(isequal(size(delta), size(layer.output)));
            layer.delta = vl_nnpool(layer.input, layer.poolSize, delta, 'stride', layer.stride, 'pad', layer.pad, 'method', layer.method);
        end
    end   
end

