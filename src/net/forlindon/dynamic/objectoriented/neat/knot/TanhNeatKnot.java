package net.forlindon.dynamic.objectoriented.neat.knot;

import net.forlindon.dynamic.objectoriented.neat.genetic.InnovationSource;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.activation.TanhTensor;

public class TanhNeatKnot extends BaseNeatKnot {
    public TanhNeatKnot(InnovationSource paramSrc, int id) {
        super(paramSrc, id);
    }

    @Override
    public Tensor getActivationTensor() {
        return new TanhTensor();
    }
}
