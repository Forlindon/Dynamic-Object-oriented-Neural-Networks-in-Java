package net.forlindon.dynamic.objectoriented.neat.experiments.tictactoe;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

public class Environment {

    public static final int GAMES = 500;

    public int sampleAction(double[] state) {
        int[] array = IntStream.range(0, state.length).filter(value -> state[value] == 0).toArray();
        return array[(int) (Math.random()*array.length)];
    }

    public void evaluate(Agent a, Agent b) {
        a.reset();
        b.reset();

        Agent start = Math.random() < 0.5 ? a : b;
        Agent second = start == a ? b : a;

        double[] stateA = start.state;
        double[] stateB = second.state;

        while (!start.done) {
            int acStart = start.getAction();
            stateA[acStart] = 1;
            stateB[acStart] = -1;
            start.turns++;
            if (isDone(stateA)) {
                start.done = true;
                second.done = true;
                break;
            }
            int acSec = second.getAction();
            stateA[acSec] = -1;
            stateB[acSec] = 1;
            second.turns++;
            if (isDone(stateA)) {
                start.done = true;
                second.done = true;
                break;
            }
        }
        double fitnessA = start.agent.a().getFitness();
        double fitnessB = second.agent.a().getFitness();
        if (hasWon(stateA, 1)) {
            fitnessA += 1;
            fitnessB -= 1;
        }
        else if (hasWon(stateA,-1)) {
            fitnessA -= 1;
            fitnessB += 1;
        }
        start.agent.a().setFitness(fitnessA);
        second.agent.a().setFitness(fitnessB);
    }

    public void evaluate(Agent agent) {
        double fitness = 0;
        for (int i = 0; i < GAMES; i++) {
            agent.reset();
            double[] state = agent.state;
            while (!agent.done) {
                int ac = agent.getAction();
                if (ac == -1) {
                    fitness--;
                }
                else {
                    state[ac] = 1;
                }
                agent.turns++;
                if (isDone(state)) {
                    agent.done = true;
                    break;
                }
                state[sampleAction(state)] = -1;
                if (isDone(state)) {
                    break;
                }
            }
            if (hasWon(state, 1)) {
                fitness += 1;
            }
            else if (hasWon(state,-1)) {
                fitness -= 1;
            }
        }
        agent.agent.a().setFitness(fitness/GAMES);
    }

    public boolean isDone(double[] state) {
        return Arrays.stream(state).filter(value -> value == 0).findAny().isEmpty()
                || hasWinner(state);
    }

    public boolean hasWinner(double[] state) {
        for (int i = 0; i < 3; i++) {
            if (state[i*3] != 0 && state[i*3] == state[i*3+1] && state[i*3] == state[i*3+2]) {
                return true;
            }
            else if (state[i] != 0 && state[i] == state[i+3] && state[i] == state[i+6]) {
                return true;
            }
        }
        return state[4] != 0 && ((state[0] == state[4] && state[0] == state[8]) || (state[2] == state[4] && state[2] == state[6]));
    }

    public boolean hasWon(double[] state, double player) {
        for (int i = 0; i < 3; i++) {
            if (state[i*3] == player && state[i*3] == state[i*3+1] && state[i*3] == state[i*3+2]) {
                return true;
            }
            else if (state[i] == player && state[i] == state[i+3] && state[i] == state[i+6]) {
                return true;
            }
        }
        return state[4] == player && ((state[0] == state[4] && state[0] == state[8]) || (state[2] == state[4] && state[2] == state[6]));
    }

    public void evaluateAll(List<Agent> agents) {
        agents.stream().parallel().forEach(this::evaluate);
    }

    public void evaluateAdvanced(List<Agent> agents) {
        agents.forEach(agent -> {
            for (int i = 0; i <GAMES; i++) {
                evaluate(agent, agents.get((int)(Math.random()*agents.size())));
            }
            agent.agent.a().setFitness(agent.agent.a().getFitness()/GAMES);
        });
    }

    public void play(Agent a, Agent b) {
        a.reset();
        b.reset();

        Agent start = Math.random() < 0.5 ? a : b;
        Agent second = start == a ? b : a;

        double[] stateA = start.state;
        double[] stateB = second.state;

        while (!start.done) {
            int acStart = start.getAction();
            stateA[acStart] = 1;
            stateB[acStart] = -1;
            start.turns++;
            if (isDone(stateA)) {
                start.done = true;
                second.done = true;
                break;
            }
            int acSec = second.getAction();
            stateA[acSec] = -1;
            stateB[acSec] = 1;
            second.turns++;
            if (isDone(stateA)) {
                start.done = true;
                second.done = true;
                break;
            }
        }

        print(a.state);
    }

    private void print(double[] state) {
        for (int i = 0; i < 3; i++) {
            System.out.printf("%s %s %s\n", translate(state[i*3]), translate(state[i*3+1]), translate(state[i*3+2]));
        }
    }

    private String translate(double d) {
        return d == 0 ? "_" : d == -1 ? "X" : "O";
    }

}
