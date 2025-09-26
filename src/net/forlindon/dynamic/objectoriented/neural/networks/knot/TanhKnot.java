package net.forlindon.dynamic.objectoriented.neural.networks.knot;

import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.activation.TanhTensor;

public class TanhKnot extends BaseKnot {
    public TanhKnot(int id) {
        super(id);
    }

    @Override
    public Tensor getActivationTensor() {
        return new TanhTensor();
    }
}
