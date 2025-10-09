package net.forlindon.dynamic.objectoriented.neural.networks.knot;

import net.forlindon.dynamic.objectoriented.neural.networks.tensor.SimpleTensor;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.activation.SimpleActivationTensor;

public class BaseKnot extends Knot{
    public BaseKnot(int id) {
        super(id);
    }

    @Override
    public Tensor getBIAS() {
        return new SimpleTensor(Math.random()*0.1);
    }

    @Override
    public Tensor getActivationTensor() {
        return new SimpleActivationTensor();
    }
}
