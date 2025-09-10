{{ config(materialized='view') }}

with base as (
    select
        event_dt::date                           as event_dt,
        event_timestamp::timestamp_ntz           as event_ts,
        user_pseudo_id::string                   as user_pseudo_id,
        lower(event_name)::string                as event_name,
        nullif(source,'')::string                as source,
        nullif(medium,'')::string                as medium,
        nullif(campaign,'')::string              as campaign,
        device_category::string                  as device_category,
        country::string                          as country,
        region::string                           as region,
        page_location::string                    as page_location,
        page_referrer::string                    as page_referrer,
        ga_session_id::number                    as ga_session_id,
        item_id::string                          as item_id,
        item_name::string                        as item_name,
        item_category::string                    as item_category,
        price::float                             as price,
        quantity::number                         as quantity
    from {{ source('raw','EVENTS_FLAT') }}
),

with_keys as (
    select
        *,
        md5(
          coalesce(to_char(event_ts,'YYYY-MM-DD HH24:MI:SS.FF6'),'') || '|' ||
          coalesce(event_name,'') || '|' ||
          coalesce(user_pseudo_id,'') || '|' ||
          coalesce(to_char(ga_session_id),'') || '|' ||
          coalesce(item_id,'')
        ) as row_key
    from base
),

dedup as (
    select
        with_keys.*,
        row_number() over (
          partition by row_key
          order by event_ts desc
        ) as rn
    from with_keys
)

select
  event_dt,
  event_ts,
  user_pseudo_id,
  event_name,
  source,
  medium,
  campaign,
  device_category,
  country,
  region,
  page_location,
  page_referrer,
  ga_session_id,
  item_id,
  item_name,
  item_category,
  price,
  quantity,
  row_key
from dedup
where rn = 1