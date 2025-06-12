**DataFair Portal Installation Manual**

1. **Multipass installation (tool for having a virtual machine without exposing our hardware resources to the test software):**
   1. **Go to the website https://canonical.com/multipass/install**
   2. **Execute the different commands according to your ecosystem**
   3. **Connect to your virtual machine with the command multipass shell <name_vm>**

2. **RKE2 cluster installation**
   1. **Type this command: curl -sfL https://get.rke2.io | sh -**

3. **Helm installation (tool for installing the project and all its DataFair components)**
   1. **Go to the website https://helm.sh/**
   2. **Execute the command according to your ecosystem**

4. **Please execute all commands in order:**
   1. **systemctl enable rke2-server.service**
   2. **systemctl start rke2-server.service**
   3. **sudo cp /etc/rancher/rke2/rke2.yaml /home/ubuntu/.kube/config**
   4. **sudo chown -R ubuntu: /home/ubuntu/.kube**
   5. **kubectl apply -f https://raw.githubusercontent.com/longhorn/longhorn/v1.8.1/deploy/longhorn.yaml**
   6. **vi values.yaml**
   7. **Paste this code and save ("hostname -I" to find the IP):**

```yaml
public_address: <Ip_Public>
scheme: http # Or https
ingress:
  enabled: false
proxy:
  enabled: true
  port: "32000"
```

8. **helm upgrade --install datafair data354-helm/data-fair -f values.yaml**
9. **kubectl get pods -A** (to verify if all components are running, wait even if there are errors, they will eventually restart)
10. **Once everything is good, you can go to a browser and enter: http://<Ip_Public>:32000/**

The Data354 DataFair portal repository: https://github.com/data354/data-fair-chart