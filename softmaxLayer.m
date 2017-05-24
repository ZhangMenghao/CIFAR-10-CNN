classdef softmaxLayer
    %TEMP Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        prob;
        pred;
        loss;
        delta;
        accuracy;
    end
    
    methods 
        function layer = forward(layer, input)%change the annotation would get the other error function
            %
            %cross entropy error
            layer.prob=exp(input);
            colsum=sum(layer.prob);
            for i=1:1:10
                layer.prob(i,:)=layer.prob(i,:)./colsum;
            end
            %}
            
            %least square error
            %layer.prob=1./(1+exp(-input));
            
        end
        
        function layer = backward(layer, label)%change the annotation would get the other error function
            %
            %cross entropy error
            layer.loss=0;
            layer.accuracy=0;
            layer.pred=[];
            for i=1:1:100
                temp=zeros(10,1);
                temp(label(i))=1;
                layer.pred=[layer.pred temp];
                layer.loss=layer.loss-sum(layer.pred(:,i).*log(layer.prob(:,i)));
                num=find(layer.prob(:,i)==max(layer.prob(:,i)));
                if (num==label(i))
                    layer.accuracy=layer.accuracy+1;
                end
            end
            layer.delta=layer.prob-layer.pred;
            layer.accuracy=layer.accuracy/100;
            %}
            
            %{
            %least square error
            layer.loss=0;
            layer.accuracy=0;
            layer.pred=[];
            for i=1:1:100
                temp=zeros(10,1);
                temp(label(i))=1;
                layer.pred=[layer.pred temp];
                layer.loss=layer.loss+sum((layer.prob(:,i)-layer.pred(:,i)).^2)/2;
                num=find(layer.prob(:,i)==max(layer.prob(:,i)));
                if (num==label(i))
                    layer.accuracy=layer.accuracy+1;
                end
            end
            layer.delta=(layer.prob-layer.pred).*layer.prob.*(1-layer.prob);
            layer.accuracy=layer.accuracy/100;
            %}
        end
    end
end

