#!/bin/bash
#
# Neo4j Docker 启动脚本
# 用于快速启动Neo4j数据库服务
#

echo "=========================================="
echo "启动 Neo4j Docker 容器"
echo "=========================================="

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo "错误: Docker 未安装，请先安装 Docker"
    echo "下载地址: https://www.docker.com/get-started"
    exit 1
fi

# 检查是否已有 neo4j 容器
if docker ps -a | grep -q "neo4j"; then
    echo "发现已存在的 neo4j 容器"
    echo "是否要删除并重新创建？(y/n)"
    read -r answer
    if [ "$answer" = "y" ]; then
        echo "正在停止并删除旧容器..."
        docker stop neo4j 2>/dev/null
        docker rm neo4j 2>/dev/null
    else
        echo "尝试启动现有容器..."
        docker start neo4j
        exit 0
    fi
fi

# 启动 Neo4j 容器
echo "正在启动 Neo4j Docker 容器..."
docker run -d \
    --name neo4j \
    -p 7474:7474 \
    -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/changeme \
    -e NEO4J_PLUGINS='["apoc"]' \
    -v $(pwd)/data/neo4j:/data \
    neo4j:latest

# 等待服务启动
echo "等待 Neo4j 启动..."
sleep 10

# 检查容器状态
if docker ps | grep -q "neo4j"; then
    echo ""
    echo "=========================================="
    echo "✓ Neo4j 启动成功！"
    echo "=========================================="
    echo ""
    echo "访问信息:"
    echo "  Web 界面: http://localhost:7474"
    echo "  Bolt 接口: bolt://localhost:7687"
    echo ""
    echo "登录凭证:"
    echo "  用户名: neo4j"
    echo "  密码: changeme"
    echo ""
    echo "停止命令:"
    echo "  docker stop neo4j"
    echo ""
    echo "查看日志:"
    echo "  docker logs neo4j"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ Neo4j 启动失败"
    echo "=========================================="
    echo ""
    echo "请检查 Docker 日志:"
    echo "  docker logs neo4j"
    echo ""
    exit 1
fi
