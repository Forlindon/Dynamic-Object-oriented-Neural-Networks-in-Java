package net.forlindon.dynamic.objectoriented.neural.networks.tensor.loss;

import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;

public class BCELoss extends Tensor {

    @Override
    public void activate(Tensor... args) {
        double y = Math.max(1e-8,Math.min(1-1e-8,args[0].val));
        double target = args[1].val;
        this.val = - (target*Math.log(y) + (1-target) * Math.log(1-y));
    }

    @Override
    public void derivative(Tensor... args) {
        double y = Math.max(1e-8,Math.min(1-1e-8,args[0].val));
        double target = args[1].val;
        args[0].pushGrad( -( target / y - (1-target) / (1-y) ) );
        args[1].pushGrad(Math.log((1-y)/y));
    }
}
