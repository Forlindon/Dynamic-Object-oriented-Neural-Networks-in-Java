package net.forlindon.dynamic.objectoriented.neural.networks.knot;

import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.activation.SigTensor;

public class SigKnot extends BaseKnot {

    public SigKnot(int id) {
        super(id);
    }

    @Override
    public Tensor getActivationTensor() {
        return new SigTensor();
    }
}
