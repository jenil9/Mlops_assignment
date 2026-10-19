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
        - name: simulator
          image: movie-recsys-simulator:dev
          imagePullPolicy: IfNotPresent
          command:
            - python
            - -m
            - src.simulator.simulate
            - --mode
            - __MODE__
            - __LIMIT_ARG__
          env:
            - name: API_URL
              value: "http://nginx:80"
            - name: SIMULATOR_KEY
              valueFrom:
                secretKeyRef:
                  name: app-secrets
                  key: SIMULATOR_KEY
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
