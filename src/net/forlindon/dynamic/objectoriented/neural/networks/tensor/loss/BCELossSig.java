package net.forlindon.dynamic.objectoriented.neural.networks.tensor.loss;

import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;

public class BCELossSig extends BCELoss {

    @Override
    public void derivative(Tensor... args) {
        args[0].pushGrad(args[0].val-args[1].val);
    }
}
