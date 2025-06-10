package com.bitus.api;

import org.springframework.web.bind.annotation.*;
import org.springframework.http.ResponseEntity;

@RestController
public class PredictController {
    private final ONNXModel model;

    public PredictController() throws Exception {
        this.model = new ONNXModel("best_model.onnx");
    }

    @PostMapping("/predict_behavior")
    public ResponseEntity<PredictionResponse> predict(@RequestBody SequenceRequest request) {
        try {
            PredictionResult result = model.predict(request.sequence);
            return ResponseEntity.ok(new PredictionResponse(result));
        } catch (Exception e) {
            return ResponseEntity.internalServerError().build();
        }
    }

    // Request DTO
    static class SequenceRequest {
        public long[] sequence;
    }

    // Response DTO
    static class PredictionResponse {
        public int prediction;
        public float[] probabilities;
        public String[] labels = {"view", "cart", "purchase"};

        public PredictionResponse(PredictionResult result) {
            this.prediction = result.predicted;
            this.probabilities = result.probabilities;
        }
    }
}
