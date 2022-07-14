DELETE
from valuesFinHistory
where rowid in (select rowid
                from (select rowid,
                             row_number() over (
        partition by dateValue, namesId
        -- order by some_expression
        ) as n
                      from valuesFinHistory)
                where n > 1);


DELETE
from financialData
where rowid in (select rowid
                from (select rowid,
                             row_number() over (
        partition by dateValue, namesId
        -- order by some_expression
        ) as n
                      from financialData)
                where n > 1);


DELETE
from namesCompanies
where rowid in (select rowid
                from (select rowid,
                             row_number() over (
        partition by ticker
        -- order by some_expression
        ) as n
                      from namesCompanies)
                where n > 1);