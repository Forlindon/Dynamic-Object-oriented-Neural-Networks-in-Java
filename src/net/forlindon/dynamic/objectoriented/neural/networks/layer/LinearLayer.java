package net.forlindon.dynamic.objectoriented.neural.networks.layer;

import net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot;

import java.util.function.Function;

public class LinearLayer extends BaseLayer {

    public LinearLayer(int n, Function<Integer, Knot> factory, int id) {
        super(id);
        for (int i = 0; i < n; i++) {
            this.add(factory);
        }
    }

}
