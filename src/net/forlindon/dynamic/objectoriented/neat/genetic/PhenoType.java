package net.forlindon.dynamic.objectoriented.neat.genetic;

import net.forlindon.dynamic.objectoriented.neat.layer.BaseNeatLayer;
import net.forlindon.dynamic.objectoriented.neural.networks.layer.Layer;

public class PhenoType {

    BaseNeatLayer net;

    public PhenoType(BaseNeatLayer net) {
        this.net = net;
    }

    public void forward(double[] in, double[] out) {
        this.net.forward(in);
        this.net.readValues(out);
        this.net.clean();
    }

    @Override
    public String toString() {
        return "PhenoType{" +
                "net=" + net +
                '}';
    }

    public Layer get() {
        return this.net;
    }
}
