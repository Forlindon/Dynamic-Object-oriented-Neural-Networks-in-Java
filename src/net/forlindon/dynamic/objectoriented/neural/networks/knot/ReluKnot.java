package net.forlindon.dynamic.objectoriented.neural.networks.knot;

import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.activation.ReluTensor;

public class ReluKnot extends Knot {

    public ReluKnot(int id) {
        super(id);
    }

    @Override
    public Tensor getActivationTensor() {
        return new ReluTensor();
    }
}
