# SQL: Structured Query Language
___

## 1. SELECT

Generic query.

```mysql
SELECT m.movies as 'movie_names',
       m.year
FROM movies m
WHERE m.movies LIKE 'The %'
      AND year >= 1999
      AND imdb_rating IS NOT NULL
ORDER BY year [ASC DESC]; # ASC by default
```

Generic query.

```mysql
SELECT *
FROM movies
WHERE name BETWEEN 'A' AND 'C'; # C not included
LIMIT 20;
```

Calculate how many rows are in a table.

```mysql
SELECT COUNT(*)
FROM movies;
```

### 1.1. Aggregation functions

- COUNT(): return the number of rows
- SUM(): cumulate the values
- AVG(): return the average of the group
- MIN/MAX(): smallest/largest values
- ROUND(column, n-integer): it rounds the values in the column to the number of decimal places specified by the integer.

### 1.2. Case (conditionals logic)

```mysql
SELECT name,
       CASE
         WHEN imdb_rating > 8 THEN 'Fantastic'
         WHEN imdb_rating > 6 THEN 'Poorly Received'
         ELSE 'Avoid at All Costs'
       END AS 'review'
FROM movies;
```

### 1.3. Wildcards

- %: Represents zero or more characters --> 'The %': The Avengers
- _: Represents a single character --> 'Se_en': Seven, Se7en
- []: Represents any single character within the brackets --> 'h[ao]t': hat, hot 

## 2. GROUP BY

```mysql
SELECT genre, COUNT(*)
FROM movies
GROUP BY genre;
```

With column reference:

```mysql
SELECT category, 
       price,
       AVG(downloads)
FROM fake_apps
GROUP BY category,
         price;
```
```mysql
SELECT category, 
       price,
       AVG(downloads)
FROM fake_apps
GROUP BY 1,
         2;
```

Limit the outputs of the group by statement:

```mysql
SELECT price, 
       ROUND(AVG(downloads)),
       COUNT(*)
FROM fake_apps
GROUP BY price
HAVING COUNT(*) > 10;
```

## 3. JOIN