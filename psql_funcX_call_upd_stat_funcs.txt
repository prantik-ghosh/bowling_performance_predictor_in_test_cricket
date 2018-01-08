CREATE OR REPLACE FUNCTION call_upd_stat_functions(v_yr_from in int, v_yr_to in int) RETURNS void AS $func$
/*
--
-- For each year of the given year range, the function calls each of
-- the individual update stat functions to generate statistics.
--
*/
DECLARE
  v_year  int;

BEGIN
  FOR v_year IN v_yr_from..v_yr_to LOOP
    EXECUTE update_bowler_stat(v_year);
    EXECUTE update_bowler_oppo_stat(v_year);
    EXECUTE update_bowler_home_adv_stat(v_year);
    EXECUTE update_oppo_bowltyp_stat(v_year);
    EXECUTE update_ground_bowltyp_stat(v_year);
  END LOOP;
END;
$func$ LANGUAGE plpgsql;





select call_upd_stat_functions(2005,2017);
