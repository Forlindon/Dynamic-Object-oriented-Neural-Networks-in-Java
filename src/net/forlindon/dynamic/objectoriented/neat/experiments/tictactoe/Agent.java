package net.forlindon.dynamic.objectoriented.neat.experiments.tictactoe;

import net.forlindon.dynamic.objectoriented.neat.experiments.Pair;
import net.forlindon.dynamic.objectoriented.neat.genetic.Genome;
import net.forlindon.dynamic.objectoriented.neat.genetic.PhenoType;
import net.forlindon.dynamic.objectoriented.neat.genetic.PhenoTypeBuilder;

import java.util.Arrays;

public class Agent {

    int turns = 0;
    double[] state = new double[9];
    double[] out = new double[9];
    Pair<Genome, PhenoType> agent;
    boolean done = false;

    public Agent(Genome genome, PhenoTypeBuilder phenoTypeBuilder) {
        this.agent = new Pair<>(genome, phenoTypeBuilder.build(genome));
    }

    public void reset() {
        this.done = false;
        this.agent.a().setFitness(0);
        Arrays.fill(state,0);
        this.turns = 0;
    }

    public int getAction() {

        agent.b().forward(state,out);
        softmax(out);

        for (int i = 0; i < out.length; i++) {
            if (state[i] != 0) out[i] = 0;
        }

        return choose();
    }

    public int choose() {
        int idx = 0;
        double max = out[0];
        for (int i = 1; i < this.out.length; i++) {
            if (out[i] > max) {
                idx = i;
                max = out[i];
            }
        }
        if (state[idx] != 0) idx = -1;
        return idx;
    }

    private static void softmax(double[] array) {
        double sum = Math.max(Arrays.stream(array).map(Math::exp).sum(), 1e-5);
        for (int i = 0; i < array.length; i++) {
            array[i] = Math.exp(array[i]) / sum;
        }
    }

}
