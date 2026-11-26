# Deploy: Amazon ECS Express Mode (Fargate, arm64)

The API runs as one Fargate task behind an HTTPS Application Load Balancer, in
**ap-southeast-1 (Singapore)**, the same region as the Supabase database.

```
user → ALB (HTTPS, checks /health) → Fargate task (Graviton, 1 vCPU / 3 GB)
          container: FastAPI + fastembed (question embeddings) → Supabase pgvector → Groq (LLM)
          secrets: Secrets Manager → env vars at start (task execution role)
          logs: CloudWatch Logs /aws/ecs/bsp/bsp-api-*
```

Cost: about $1.30/day while running (Fargate + ALB + public IPv4). **Delete after demos.**

## Files

| File | What it is |
| --- | --- |
| `trust-ecs-tasks.json` | Trust policy: `bspTaskExecutionRole` may be assumed by ECS tasks |
| `trust-ecs.json` | Trust policy: `bspEcsInfrastructureRole` may be assumed by ECS (builds the ALB etc.) |
| `execution-secret-policy.json` | Lets the execution role read **only** the `bsp-circular-copilot/prod` secret |
| `service.json` | The service: image tag, port 8080, `/health`, secrets, CPU/memory, ARM64, 1 task |
| `insights-ask-stats.query` | CloudWatch Logs Insights query: questions, latency, tokens by refused |

## One-time setup (already done)

1. Secrets Manager (console): secret `bsp-circular-copilot/prod` with keys `GROQ_API_KEY`, `DATABASE_URL`.
2. ECR: `aws ecr create-repository --repository-name bsp-circular-copilot --image-tag-mutability IMMUTABLE --image-scanning-configuration scanOnPush=true`
3. IAM roles (see the JSON files above):
   ```bash
   aws iam create-role --role-name bspTaskExecutionRole --assume-role-policy-document file://deploy/trust-ecs-tasks.json
   aws iam attach-role-policy --role-name bspTaskExecutionRole --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
   aws iam put-role-policy --role-name bspTaskExecutionRole --policy-name read-bsp-secret --policy-document file://deploy/execution-secret-policy.json
   aws iam create-role --role-name bspEcsInfrastructureRole --assume-role-policy-document file://deploy/trust-ecs.json
   aws iam attach-role-policy --role-name bspEcsInfrastructureRole --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSInfrastructureRoleforExpressGatewayServices
   aws ecs create-cluster --cluster-name bsp
   ```

## Push a new image

```bash
REGISTRY=969674340441.dkr.ecr.ap-southeast-1.amazonaws.com
TAG=$(git rev-parse --short HEAD)
aws ecr get-login-password | docker login --username AWS --password-stdin $REGISTRY
docker build --platform linux/arm64 -t $REGISTRY/bsp-circular-copilot:$TAG .
docker push $REGISTRY/bsp-circular-copilot:$TAG
```
Then put the new tag in `service.json` (`primaryContainer.image`).

## Create the service (demo on)

```bash
aws ecs create-express-gateway-service --cli-input-json file://deploy/service.json --monitor-resources DEPLOYMENT
```
The public URL is the `IngressPath` in the output (about 5 minutes until it answers).

## Delete the service (demo off)

```bash
aws ecs delete-express-gateway-service --service-arn arn:aws:ecs:ap-southeast-1:969674340441:service/bsp/bsp-api
```
This removes the ALB, certificate, target groups, security groups and tasks. The ECR image, the
secret, the roles, the cluster and the log group stay (free or a few cents per month).

## Logs

```bash
aws logs tail /aws/ecs/bsp/bsp-api-6c33 --since 1h --filter-pattern '"ask latency_ms"'
```
Logs are kept for 14 days (`aws logs put-retention-policy … --retention-in-days 14`).
The log group name gets a new suffix if the service is recreated: check Log Management.
