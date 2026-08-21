package org.example;

import java.util.Random;

public class Perceptron {

    private double[] weights;  // W1, W2, W3
    private double theta;      // threshold - fixed after init
    private double alpha;      // learning rate

    public Perceptron(double alpha) {
        this.alpha = alpha;
        Random rand = new Random();
        weights = new double[3];
        for (int i = 0; i < 3; i++)
            weights[i] = rand.nextDouble() - 0.5;  // [-0.5, 0.5]
        theta = rand.nextDouble() - 0.5;
    }

    // X = x1*W1 + x2*W2 + x3*W3 - theta
    // output = (X >= 0) ? 1 : 0
    public int predict(double[] x) {
        double X = 0;
        for (int i = 0; i < 3; i++)
            X += x[i] * weights[i];
        X -= theta;
        return (X >= 0) ? 1 : 0;
    }

    // update weights only when prediction is wrong
    public void update(double[] x, int label) {
        int pred = predict(x);
        int delta = label - pred;
        if (delta != 0) {
            for (int i = 0; i < 3; i++)
                weights[i] += alpha * delta * x[i];
            // theta
        }
    }

    public int[] fit(double[][] X, int[] y, int maxEpochs) {
        int[] errorsPerEpoch = new int[maxEpochs];
        for (int e = 0; e < maxEpochs; e++) {
            int errors = 0;
            for (int i = 0; i < X.length; i++) {
                int pred = predict(X[i]);
                if (pred != y[i]) {
                    update(X[i], y[i]);
                    errors++;
                }
            }
            errorsPerEpoch[e] = errors;
            if (errors == 0) {
                System.out.println("Converged at epoch " + (e + 1));
                return java.util.Arrays.copyOf(errorsPerEpoch, e + 1);
            }
        }
        return errorsPerEpoch;
    }

    public double accuracy(double[][] X, int[] y) {
        int correct = 0;
        for (int i = 0; i < X.length; i++)
            if (predict(X[i]) == y[i]) correct++;
        return (double) correct / X.length;
    }

    // normalize inputs before calling predict/update
    // x1 = moisture / 100
    // x2 = lastWatered / 48
    // x3 = plantType / 2
    public static double[] normalize(int moisture, int lastWatered, int plantType) {
        return new double[]{ moisture / 100.0, lastWatered / 48.0, plantType / 2.0 };
    }

    public double[] getWeights() { return weights; }
    public double getTheta()     { return theta; }
    public double getAlpha()     { return alpha; }
}