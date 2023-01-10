# SQL: Structured Query Language
___

## 1. Select

Generic query:

```mysql
SELECT m.movies as 'movie_names',
       m.year
FROM movies as m
WHERE m.movies LIKE 'The %'
  AND year >= 1999
  AND imdb_rating IS NOT NULL
ORDER BY year [ASC DESC]; # ASC by default
```

Generic query:

```mysql
SELECT *
FROM movies
WHERE name BETWEEN 'A' AND 'C'; -- C not included
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
SELECT 
    name,
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

## 2. Group by

```mysql
SELECT genre, COUNT(*)
FROM movies
GROUP BY genre;
```

With column reference:

```mysql
SELECT
    category, 
    price,
    AVG(downloads)
FROM fake_apps
GROUP BY category, price;
```
```mysql
SELECT 
    category, 
    price,
    AVG(downloads)
FROM fake_apps
GROUP BY 1, 2;
```

Limit the outputs of the group by statement:

```mysql
SELECT
    price, 
    ROUND(AVG(downloads)),
    COUNT(*)
FROM fake_apps
GROUP BY price
HAVING COUNT(*) > 10;
```

## 3. Union

UNION ALL does not analyze if exists any duplicates between tables.

```mysql
SELECT *
FROM table1
--
UNION ALL
--
SELECT *
FROM table2;
```

UNION eliminates duplicates once the UNION has been done.

```mysql
SELECT *
FROM table1
--
UNION
--
SELECT *
FROM table2;
```

## 4. Join

Types of JOIN:

- INNER: only match between keys
- LEFT: all rows from left join with all the matching rows from the righ table
- CROSS JOIN: all rows of one table with all rows of another table

```mysql
SELECT t1.order_id,
       t2.customer_name
FROM orders as t1
LEFT JOIN customers as t2 ON t1.customer_id = t2.customer_id
WHERE t2.customer_id IS NULL;
```

```mysql
SELECT *
FROM left_table
INNER JOIN right_table ON left_table.id = right_table.id
INNER JOIN another_table ON left_table.id = another_table.id;
```

## 5. Subqueries

A subquery is usually added within the WHERE Clause of another SQL SELECT statement.

```mysql
SELECT ProductName
FROM Product 
WHERE Id IN (
    SELECT ProductId 
    FROM OrderItem
    WHERE Quantity > 100
    );
```

### 6. Query-optimization advices

- https://gabrielmorrissa.blogspot.com/2017/02/buenas-practicas-para-elaborar.html
