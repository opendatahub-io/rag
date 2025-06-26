# Deploying llamastack playground ui on OpenShift

## Clone llamastack

```
git clone https://github.com/meta-llama/llama-stack.git
cd llama_stack/distribution/ui
```

## Building image

```
oc import-image ubi8/python-312 --from=registry.redhat.io/ubi8/python-312 --confirm
oc new-app --name=playground . --image-stream="python-312" --context-dir=llama_stack/distribution/ui
```

## Configuring, patching and deploying
Set llamastack endpoint route
```
LS_ROUTE=$(oc get route llamastack -ojsonpath={.spec.host})
oc set env deployment/playground LLAMA_STACK_ENDPOINT=http://$LS_ROUTE
```

Change:
- entrypoint to `streamlit` as openshift python image will use app.py only instead.
- port to 8501 as python image uses 8080 instead

```
oc patch deployment playground  -p '{"spec":{"template":{"spec":{"containers":[{"name":"playground","command":["streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0"],"ports":[{"containerPort":8501,"protocol":"TCP"}]}]}}}}'
oc patch svc playground  -p '{"spec":{"ports":[{"port":8501,"targetPort":8501,"protocol":"TCP","name":"http"}]}}'
```

Expose `service` through a route and patch it
```
oc expose svc playground
oc patch route playground -p '{"spec":{"port":{"targetPort":"http"}}}'
```


## Getting manifests

```
oc eksporter is                                    > 01-is.yaml
oc eksporter bc playground --drop spec.triggers    > 02-bc.yaml
oc eksporter deployment playground                 > 03-dc.yaml
oc eksporter svc playground --drop spec.clusterIPs > 04-svc.yaml
oc eksporter route playground --drop spec.host     > 05-route.yaml
```

## Installing everything using manifests

```
oc create -f .
```

