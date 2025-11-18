package net.forlindon.dynamic.objectoriented.neat.knot;

import net.forlindon.dynamic.objectoriented.neat.genetic.InnovationSource;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.Tensor;
import net.forlindon.dynamic.objectoriented.neural.networks.tensor.activation.SigTensor;

public class SigNeatKnot extends BaseNeatKnot {
    public SigNeatKnot(InnovationSource paramSrc, int id) {
        super(paramSrc, id);
    }

    @Override
    public Tensor getActivationTensor() {
        return new SigTensor();
    }
}
