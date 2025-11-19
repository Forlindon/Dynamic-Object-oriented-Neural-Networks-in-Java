package net.forlindon.dynamic.objectoriented.neural.networks.knot;

import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.activation.ParametricRelu;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.activation.ReluTensor;

public class ParametricReluKnot extends BaseKnot {

    double alpha;

    public ParametricReluKnot(int id, double alpha) {
        super(id);
        this.alpha = alpha;
    }

    @Override
    public Tensor getActivationTensor() {
        return new ParametricRelu(this.alpha);
    }


}
