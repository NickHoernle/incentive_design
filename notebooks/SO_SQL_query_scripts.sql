/*
Script to get the most commonly occurring badges on SO
*/

SELECT count(Id) as 'count', Name FROM Badges
WHERE
  Class=1 AND TagBased = 1
GROUP BY Name
Having COUNT(Id) > 25
ORDER BY COUNT(Id) DESC;





/*
Get the Users who achieved the most commonly occurring Badges
*/
SELECT b1.UserId, b1.Name, b1.Date FROM Badges as b1
INNER JOIN (
      SELECT count(Id) as 'count', Name FROM Badges
      WHERE
        Class=1 AND TagBased = 1
      GROUP BY Name
      Having COUNT(Id) > 25
    ) b2
  ON (
    b1.Name = b2.Name AND
    b1.TagBased = 1 AND
    b1.Class = 1
  )
GROUP BY b1.UserId, b1.Name, b1.Date;


/*
UserIds who achieved the most Badges
*/
SELECT b1.UserId, count(b1.Name) as 'badge_count' FROM Badges as b1
INNER JOIN (
      SELECT count(Id) as 'count', Name FROM Badges
      WHERE
        Class=1 AND TagBased = 1
      GROUP BY Name
      Having COUNT(Id) > 25
    ) b2
  ON (
    b1.Name = b2.Name AND
    b1.TagBased = 1 AND
    b1.Class = 1
  )
GROUP BY b1.UserId
HAVING count(b1.UserId) > 5;


/*
Posts from Users who achieved the most Badges
*/
SELECT Id, PostTypeId, CreationDate, Score, Tags, OwnerUserId
FROM Posts as p
INNER JOIN (
        SELECT b1.UserId, count(b1.Name) as 'badge_count' FROM Badges as b1
        INNER JOIN (
              SELECT count(Id) as 'count', Name FROM Badges
              WHERE
                Class=1 AND TagBased = 1
              GROUP BY Name
              Having COUNT(Id) > 25
            ) b2
          ON (
            b1.Name = b2.Name AND
            b1.TagBased = 1 AND
            b1.Class = 1
          )
        GROUP BY b1.UserId
        HAVING count(b1.UserId) > 5
    ) users
    ON (
      p.PostTypeId = 2 AND
      p.OwnerUserId = users.UserId
    )



/* Posts from Users who got the most badges */
SELECT answers.Id, answers.PostTypeId,
       answers.CreationDate, answers.Score,
       answers.OwnerUserId, answers.ParentId, questions.tags
FROM Posts as answers
INNER JOIN (
        SELECT b1.UserId, count(b1.Name) as 'badge_count' FROM Badges as b1
        INNER JOIN (
              SELECT count(Id) as 'count', Name FROM Badges
              WHERE
                Class=2 AND TagBased = 1
              GROUP BY Name
              Having COUNT(Id) > 100
            ) b2
          ON (
            b1.Name = b2.Name AND
            b1.TagBased = 1 AND
            b1.Class = 2
          )
        GROUP BY b1.UserId
        HAVING count(b1.UserId) > 10
    ) users
    ON (
      answers.PostTypeId = 2 AND
      answers.OwnerUserId = users.UserId
    )
INNER JOIN Posts as questions ON questions.Id = answers.ParentId




CREATE TABLE posts (
    Id INT NOT NULL PRIMARY KEY,
    PostTypeId SMALLINT,
    AcceptedAnswerId INT,
    ParentId INT,
    Score INT NULL,
    ViewCount INT NULL,
    Body text NULL,
    OwnerUserId INT NOT NULL,
    OwnerDisplayName VARCHAR(256) NULL,
    LastEditorUserId INT,
    LastEditDate DATETIME,
    LastActivityDate DATETIME,
    Title varchar(256) NULL,
    Tags VARCHAR(256),
    AnswerCount INT NOT NULL DEFAULT 0,
    CommentCount INT NOT NULL DEFAULT 0,
    FavoriteCount INT NOT NULL DEFAULT 0,
    ClosedDate DATETIME,
    CreationDate DATETIME
);
CREATE TABLE users (
    Id INT NOT NULL PRIMARY KEY,
    Reputation INT NOT NULL,
    CreationDate DATETIME,
    DisplayName VARCHAR(50) NULL,
    LastAccessDate  DATETIME,
    Views INT DEFAULT 0,
    WebsiteUrl VARCHAR(256) NULL,
    Location VARCHAR(256) NULL,
    AboutMe TEXT NULL,
    Age INT,
    UpVotes INT,
    DownVotes INT,
    EmailHash VARCHAR(32)
);
CREATE TABLE comments (
    Id INT NOT NULL PRIMARY KEY,
    PostId INT NOT NULL,
    Score INT NOT NULL DEFAULT 0,
    Text TEXT,
    CreationDate DATETIME,
    UserId INT NOT NULL
);
create table badges (
  Id INT NOT NULL PRIMARY KEY,
  UserId INT,
  Name VARCHAR(50),
  Date DATETIME,
  Class SMALLINT NOT NULL,
  TagBased Boolean NOT NULL
);
CREATE TABLE posthistory (
    Id INT NOT NULL PRIMARY KEY,
    PostHistoryTypeId SMALLINT NOT NULL,
    PostId INT NOT NULL,
    RevisionGUID VARCHAR(36),
    CreationDate DATETIME,
    UserId INT NOT NULL,
    Comment TEXT,
    Text TEXT
);
CREATE TABLE votes (
    Id INT NOT NULL PRIMARY KEY,
    PostId INT NOT NULL,
    VoteTypeId SMALLINT,
    CreationDate DATETIME,
    UserId INT,
    BountyAmount INT
);

LOAD XML LOCAL INFILE '~/Downloads/stack_overflow/Badges.xml'
INTO TABLE badges
ROWS IDENTIFIED BY '<row>';

LOAD XML LOCAL INFILE '~/Downloads/stack_overflow/Users.xml'
INTO TABLE users
ROWS IDENTIFIED BY '<row>';

LOAD XML LOCAL INFILE '~/Downloads/stack_overflow/Comments.xml'
INTO TABLE comments
ROWS IDENTIFIED BY '<row>';

LOAD XML LOCAL INFILE '~/Downloads/stack_overflow/Votes.xml'
INTO TABLE votes
ROWS IDENTIFIED BY '<row>';




LOAD XML LOCAL INFILE '~/Downloads/stack_overflow/Posts.xml'
INTO TABLE posts
ROWS IDENTIFIED BY '<row>';

LOAD XML LOCAL INFILE '~/Downloads/stack_overflow/PostHistory.xml'
INTO TABLE post_history
ROWS IDENTIFIED BY '<row>';


create index badges_idx_1 on badges(UserId);

create index comments_idx_1 on comments(PostId);
create index comments_idx_2 on comments(UserId);

create index post_history_idx_1 on post_history(PostId);
create index post_history_idx_2 on post_history(UserId);

create index posts_idx_1 on posts(AcceptedAnswerId);
create index posts_idx_2 on posts(ParentId);
create index posts_idx_3 on posts(OwnerUserId);
create index posts_idx_4 on posts(LastEditorUserId);

create index votes_idx_1 on votes(PostId);
