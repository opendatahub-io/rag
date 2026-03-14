oc eksporter is                                    > 01-is.yaml
oc eksporter bc playground --drop spec.triggers    > 02-bc.yaml
oc eksporter deployment playground                 > 03-dc.yaml
oc eksporter svc playground --drop spec.clusterIPs > 04-svc.yaml
oc eksporter route playground --drop spec.host     > 05-route.yaml


