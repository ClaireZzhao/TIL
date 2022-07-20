# SQL 연습문제

1. 2017년 각 에든버러 선거구에서 승리한 정당을 구하는 예시(from 절에 서브쿼리 +  window function 사용)

<aside>
👉 승리한 정당: 득표수가 가장 높은 정당

</aside>

```sql
select constituency, party 
from (select constituency, 
             party, 
             votes, 
             rank()over(partition by constituency order by votes desc) rk 
             from ge 
             where yr = 2017 and 
             constituency between 'S14000021' and 'S14000026') as new 
where new.rk = 1
```

1. 모든 나라의 인구수가 25000000보다 적은 대륙을 구하는 예시 (not in 사용)

```sql
select name, continent, population from world
where continent not in (select distinct continent 
                        from world 
                        where population > 25000000)
```

1. 프랑스와 이탈리아의 코로나 신규감연자수 구하는 예시 (서브쿼리 + window function 사용)

```sql
select name, date, 每天新增治愈人数, rank()over(partition by name order by 每天新增治愈人数 desc) rk 
from (select name, date_format(whn, '%Y年%m月%d日') date, (recovered - lag(recovered, 1) over(partition by name order by whn)) 每天新增治愈人数 
      from covid where name in ('France', 'Italy')) re
order by rk
```