package net.forlindon.dynamic.objectoriented.neat.experiments.cardpole;

public class CardPoleEnvironment {

    double[] out = new double[1];
    final double g = 9.8;    // gravity
    final double m = 0.1;    // mass of the pole
    final double M = 1.0;    // mass of the cart
    final double l = 0.5;    // half the length of the pole
    final double tau = 0.02; // time step (in seconds)
    double maxAngle = Math.PI / 2 - 0.01;

    public static final int maxEpisodes = 2000;

    public void runAgent(Agent agent) {

        if (agent.done) return;
        if (agent.episodes++ >= maxEpisodes) agent.done = true;

        double[] state = agent.state(); // this.theta,this.thetaDot,this.x,this.xDot
        double theta = state[0];
        double thetaDot = state[1];
        double x = state[2];
        double xDot = state[3];

        double costheta = Math.cos(theta);
        double sintheta = Math.sin(theta);

        agent.action(out);
        double forceMag = out[0];
        if (forceMag < -0.5) forceMag = -10;
        else if (forceMag > 0.5) forceMag = 10;
        else forceMag = 0;

        double temp = (forceMag + m * l * thetaDot * thetaDot * sintheta) / (M + m);
        double thetaAcc = (g * sintheta - costheta * temp) /
                (l * (4.0/3.0 - m * costheta * costheta / (M + m)));
        double xAcc = temp - m * l * thetaAcc * costheta / (M + m);

        x += tau * xDot;
        xDot += tau * xAcc;
        theta += tau * thetaDot;
        thetaDot += tau * thetaAcc;

        if (theta > maxAngle) {
            theta = maxAngle;
            thetaDot = 0;
            xDot = 0;
            agent.done = true;
        } else if (theta < -maxAngle) {
            theta = -maxAngle;
            thetaDot = 0;
            xDot = 0;
            agent.done = true;
        }
        agent.setState(theta,thetaDot,x,xDot);
    }

}
