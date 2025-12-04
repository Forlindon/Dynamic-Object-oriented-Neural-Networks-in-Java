package net.forlindon.dynamic.objectoriented.neat.experiments.cardpole;

import net.forlindon.dynamic.objectoriented.neat.experiments.Pair;
import net.forlindon.dynamic.objectoriented.neat.genetic.Genome;
import net.forlindon.dynamic.objectoriented.neat.genetic.PhenoType;

import java.util.Arrays;

public class Agent {

    Pair<Genome, PhenoType> agent;
    double[] state = new double[4];
    boolean done;
    int episodes;

    public Agent(Pair<Genome, PhenoType> pair) {
        this.agent = pair;
        setState(Math.random() * 0.02 - 0.01,Math.random() * 0.02 - 0.01,Math.random() * 0.02 - 0.01,Math.random() * 0.02 - 0.01);
    }

    void action(double[] out) {
        agent.b().forward(state,out);
    }

    public double[] state() {
        return state;
    }

    public void setState(double theta, double thetaDot, double x, double xDot) {
        this.state[0]=theta;
        this.state[1]=thetaDot;
        this.state[2]=x;
        this.state[3]=xDot;
    }

    public void reset() {
        this.done = false;
        this.episodes = 0;
        Arrays.fill(this.state, Math.random() * 0.02 - 0.01);
    }
}
