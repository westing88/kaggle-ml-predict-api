package com.bitus.api;

import ai.onnxruntime.*;
import java.util.*;

public class ONNXModel {
    private OrtEnvironment env;
    private OrtSession session;

    public ONNXModel(String modelPath) throws OrtException {
        env = OrtEnvironment.getEnvironment();
        session = env.createSession(modelPath, new OrtSession.SessionOptions());
    }

    public PredictionResult predict(long[] sequence) throws OrtException {
        // prepare input: shape = [1, window] (e.g., [1, 2])
        long[][] input = new long[1][sequence.length];
        input[0] = sequence;

        OnnxTensor tensor = OnnxTensor.createTensor(env, input);
        OrtSession.Result result = session.run(Collections.singletonMap("input", tensor));

        float[][] logits = (float[][]) result.get(0).getValue();

        float[] probs = softmax(logits[0]);
        int predicted = argmax(probs);

        return new PredictionResult(predicted, probs);
    }


    private float[] softmax(float[] x) {
        double max = -Double.MAX_VALUE;
        for (float f : x) {
            if (f > max) max = f;
        }
    
        double sum = 0.0;
        float[] result = new float[x.length];
    
        for (int i = 0; i < x.length; i++) {
            result[i] = (float) Math.exp(x[i] - max);
            sum += result[i];
        }
    
        for (int i = 0; i < result.length; i++) {
            result[i] /= sum;
        }
    
        return result;
    }
    

    private int argmax(float[] probs) {
        int maxIndex = 0;
        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > probs[maxIndex]) maxIndex = i;
        }
        return maxIndex;
    }
}
