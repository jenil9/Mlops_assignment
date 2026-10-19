apiVersion: batch/v1
kind: Job
metadata:
  name: __NAME__
  namespace: movie-recsys
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: trainer
          image: movie-recsys-trainer:dev
          imagePullPolicy: IfNotPresent
          command:
            - sh
            - -c
            - "__COMMAND__"
          env:
            - name: API_HEADLESS_HOST
              value: "api-headless:8000"
          envFrom:
            - secretRef:
                name: app-secrets
          volumeMounts:
            - name: data
              mountPath: /app/data/processed
              readOnly: true
            - name: params
              mountPath: /app/params.yaml
              subPath: params.yaml
      volumes:
        - name: data
          hostPath:
            path: /mnt/data/processed
            type: Directory
        - name: params
          configMap:
            name: params
