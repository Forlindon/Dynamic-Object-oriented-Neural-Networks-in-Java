package net.forlindon.dynamic.objectoriented.neural.networks.layer;

import net.forlindon.dynamic.objectoriented.neural.networks.connection.Connection;
import net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot;

import java.util.function.BiFunction;
import java.util.function.Function;

public class LinearSequentialLayer extends SequentialLayer {

    @SafeVarargs
    public LinearSequentialLayer(BiFunction<Knot,Knot, Connection> connectionFactory, Function<Integer, Layer>... factories) {
        super();
        if (factories.length < 1) return;
        this.LAYERS.add(factories[0].apply(0));
        for (int i = 1; i < factories.length; i++) {
            this.LAYERS.add(factories[i].apply(i));
            this.LAYERS.get(i-1).fullConnect(this.LAYERS.get(i), connectionFactory);
        }
    }

}
