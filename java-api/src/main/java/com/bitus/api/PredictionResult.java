package com.bitus.api;

public class PredictionResult {
    public int predicted;
    public float[] probabilities;

    public PredictionResult(int predicted, float[] probabilities) {
        this.predicted = predicted;
        this.probabilities = probabilities;
    }
}
