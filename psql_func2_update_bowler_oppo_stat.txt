CREATE OR REPLACE FUNCTION update_bowler_oppo_stat(v_year in int) RETURNS void AS $func$
/*
--
-- This function generates bowler and opponent specific statistics.
-- It calculates the ratio of average wickets taken per match by
-- this player against this opponent vs against all opponents.
--
*/
DECLARE
  bowler_oppo_list CURSOR IS
    SELECT
      distinct player, opposition
    FROM
      bowling_data
    WHERE
      year = v_year;

  v_player             text;
  v_mtchs_tot          int;
  v_wkts_tot           int;
  v_mtchs_oppo         int;
  v_wkts_oppo          int;
  v_bowler_agnst_oppo  double precision;

BEGIN
  v_player := 'XYZ';

  FOR cur_rec IN bowler_oppo_list LOOP
    IF cur_rec.player != v_player THEN
      v_player := cur_rec.player;

      SELECT
         count(1)
        ,sum(wkts)
      INTO
         v_mtchs_tot
        ,v_wkts_tot
      FROM
        bowling_data
      WHERE
            year < v_year
        and player = v_player;
    END IF;

    SELECT
       count(1)
      ,sum(wkts)
    INTO
       v_mtchs_oppo
      ,v_wkts_oppo
    FROM
      bowling_data
    WHERE
          year < v_year
      and player = v_player
      and opposition = cur_rec.opposition;

    IF (v_mtchs_oppo = 0 OR v_mtchs_tot = 0 or v_wkts_tot = 0) THEN
      v_bowler_agnst_oppo = 1;
    ELSE
      v_bowler_agnst_oppo = (cast(v_wkts_oppo as float)/v_mtchs_oppo)/(cast(v_wkts_tot as float)/v_mtchs_tot);
    END IF;

    UPDATE
       bowling_data
    SET
      bowler_agnst_oppo = v_bowler_agnst_oppo
    WHERE
          year = v_year
      and player = v_player
      and opposition = cur_rec.opposition;

  END LOOP;
END;
$func$ LANGUAGE plpgsql;

