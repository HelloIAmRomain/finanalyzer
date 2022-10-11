# Create a mysql database called newsletter then create a table called finanalyzer that has 2 columns id and email
FROM mysql
ENV MYSQL_ROOT_PASSWORD DoNotUseThisPasswordItIsNotSupposedToBeUsedThankYouForYourCooperationAndHaveAGreatDay
ENV MYSQL_DATABASE finanalyzer

COPY ./sql-scripts/ /docker-entrypoint-initdb.d/

