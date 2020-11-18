
!/bin/bash
# Bash script for pushing and putting containers to production
echo Dry run Docker
echo Please enter version number!
read version
echo your version number is $version
# Build docker containers
docker build -t in:v$version .
# Push to Googlce cloud
docker run --rm -p 6379:6379 in:v$version